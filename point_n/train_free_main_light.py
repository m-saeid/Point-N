import os
import time
import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import general_utils as gutils
from models.point_gn import PointGNCls
from models.point_nn import PointNNCls
from models.point_nn_original import Point_NN_Original
from models.point_gn_light import PointGNPCls
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import progress_bar

import sklearn.metrics as metrics



def process_data(data_loader, model, device):
    features_list, labels_list = [], []

    # Loop through the provided data loader
    for points, labels in tqdm(data_loader, leave=False):
        # points: [B, num_points, 3]
        point_features = model(points.to(device))  # [B, num_features]
        features_list.append(point_features)

        labels = labels.to(device)
        labels_list.append(labels)

    features = torch.cat(features_list, dim=0)  # [num_samples, num_features]
    features = F.normalize(features, dim=-1)
    # features = features.permute(1, 0)  # [num_features, num_samples]

    labels = torch.cat(labels_list, dim=0)  # [num_samples, 1]

    return features, labels


def get_model_and_param(args):
    if args.model == "pointgn":
        args.seed = 42
        args.init_dim = 27
        args.stage_dim = 27
        args.stages = 4
        args.k = 120

        if args.dataset == "modelnet40":
            args.sigma = 0.4

        elif args.dataset == "scanobject":
            args.sigma = 0.3

        model = PointGNPCls(
            num_points=args.num_points,
            init_dim=args.init_dim,
            stages=args.stages,
            stage_dim=args.stage_dim,
            k=args.k,
            sigma=args.sigma,
            feat_normalize=args.feat_normalize,
        )

    return model


def cal_loss(pred, gold, eps, smoothing=True):
    """Calculate cross entropy loss, apply label smoothing if needed."""

    gold = gold.contiguous().view(-1)

    if smoothing:
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction="mean")

    return loss


def process_and_evaluate(train_loader, test_loader, model, device):
    # Process training data
    start_train_time = time.time()
    train_features, train_labels = process_data(train_loader, model, device)
    train_labels = F.one_hot(train_labels).squeeze().float()
    train_time = time.time() - start_train_time

    # Process testing data
    start_test_time = time.time()
    test_features, test_labels = process_data(test_loader, model, device)
    test_time = time.time() - start_test_time

    # Calculate accuracies
    acc_cos, gamma = gutils.cosine_similarity(
        test_features, train_features, train_labels, test_labels
    )
    acc_1nn = gutils.one_nn_classification(
        test_features, train_features, train_labels, test_labels
    )

    # Return results
    return {
        "train_time": train_time,
        "test_time": test_time,
        "acc_1nn": acc_1nn,
        "acc_cos": acc_cos,
        "gamma": gamma,
    }


def main_cls():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        cudnn.benchmark = True

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    dataset_dir = os.path.join(project_root, "datasets")

    args = gutils.get_arguments()
    train_loader, test_loader = gutils.get_cls_dataloader(dataset_dir, args)

    gutils.set_seed(args.seed)

    model = get_model_and_param(args)

    model.to(device)
    start_total_time = time.time()

    criterion = cal_loss

    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()

    print("model parameters: " + str(num_params))

    args.optim = "adam"
    args.learning_rate = 0.01
    args.weight_decay = 2e-4

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, eps=1e-4
        )

    elif args.optim == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, eps=1e-4
        )

    start_epoch = 0
    args.epoch = 600
    scheduler = CosineAnnealingLR(
        optimizer, args.epoch, eta_min=1e-5, last_epoch=start_epoch - 1
    )

    best_test_acc = 0.0
    best_train_acc = 0.0
    best_test_acc_avg = 0.0
    best_train_acc_avg = 0.0
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0

    # save_args(args)

    args.eps = 0.4

    for epoch in range(start_epoch, args.epoch):
        print(
            "Epoch(%d/%s) Learning Rate %s:"
            % (epoch + 1, args.epoch, optimizer.param_groups[0]["lr"])
        )

        train_out = train(model, train_loader, optimizer, criterion, args.eps, device)
        test_out = validate(model, test_loader, criterion, args.eps, device)

        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = (
            test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        )
        best_train_acc = (
            train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        )
        best_test_acc_avg = (
            test_out["acc_avg"]
            if (test_out["acc_avg"] > best_test_acc_avg)
            else best_test_acc_avg
        )
        best_train_acc_avg = (
            train_out["acc_avg"]
            if (train_out["acc_avg"] > best_train_acc_avg)
            else best_train_acc_avg
        )
        best_test_loss = (
            test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        )
        best_train_loss = (
            train_out["loss"]
            if (train_out["loss"] < best_train_loss)
            else best_train_loss
        )

        # save_model(
        #     net, epoch, path=args.ckpt_dir, acc=test_out["acc"], is_best=is_best,
        #     best_test_acc=best_test_acc,
        #     best_train_acc=best_train_acc,
        #     best_test_acc_avg=best_test_acc_avg,
        #     best_train_acc_avg=best_train_acc_avg,
        #     best_test_loss=best_test_loss,
        #     best_train_loss=best_train_loss,
        #     optimizer=optimizer.state_dict()
        # )

        print(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s"
        )
        print(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n"
        )

    print(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    print(
        f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++"
    )
    print(
        f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++"
    )
    print(
        f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++"
    )
    print(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    print(f"++++++++" * 5)


def train(model, trainloader, optimizer, criterion, eps, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.contiguous().to(device), label.to(device).squeeze()
        # data = data.permute(0, 2, 1)
        optimizer.zero_grad()
        logits = model(data)

        loss = criterion(logits, label, eps)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100.0 * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float(
            "%.3f" % (100.0 * metrics.balanced_accuracy_score(train_true, train_pred))
        ),
        "time": time_cost,
    }


def validate(net, testloader, criterion, eps, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            # data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label, eps)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100.0 * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float(
            "%.3f" % (100.0 * metrics.balanced_accuracy_score(test_true, test_pred))
        ),
        "time": time_cost,
    }


if __name__ == "__main__":
    main_cls()
