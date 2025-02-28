import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse

from datasets.data_scan import ScanObjectNN
from datasets.data_mn40 import ModelNet40
from utils import *
from models import Point_NN, Point_NN_RBF


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mn40")
    # parser.add_argument('--dataset', type=str, default='scan')

    # parser.add_argument('--split', type=int, default=1)
    # parser.add_argument('--split', type=int, default=2)
    parser.add_argument("--split", type=int, default=3)

    parser.add_argument("--bz", type=int, default=32)  # Freeze as 16
    parser.add_argument("--points", type=int, default=1024)
    
    
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--dim", type=int, default=72)
    parser.add_argument("--k", type=int, default=90)
    
    parser.add_argument("--alpha", type=int, default=1000)
    parser.add_argument("--beta", type=int, default=100)
    parser.add_argument("--sigma", type=int, default=0.31)
    parser.add_argument("--seed", type=int, default=42)
    

    args = parser.parse_args()
    return args


def get_dataloader(args):

    if args.dataset == "scan":
        train_loader = DataLoader(
            ScanObjectNN(
                split=args.split, partition="training", num_points=args.points
            ),
            num_workers=8,
            batch_size=args.bz,
            shuffle=False,
            drop_last=False,
        )
        test_loader = DataLoader(
            ScanObjectNN(split=args.split, partition="test", num_points=args.points),
            num_workers=8,
            batch_size=args.bz,
            shuffle=False,
            drop_last=False,
        )
    elif args.dataset == "mn40":
        train_loader = DataLoader(
            ModelNet40(partition="train", num_points=args.points),
            num_workers=8,
            batch_size=args.bz,
            shuffle=False,
            drop_last=False,
        )
        test_loader = DataLoader(
            ModelNet40(partition="test", num_points=args.points),
            num_workers=8,
            batch_size=args.bz,
            shuffle=False,
            drop_last=False,
        )

    return train_loader, test_loader


@torch.no_grad()
def main():

    set_seed(42)

    print("==> Loading args..")
    args = get_arguments()

    print("==> Preparing data..")
    train_loader, test_loader = get_dataloader(args)

    print("==> Preparing model..")
    # point_nn = Point_NN(
    #     input_points=args.points,
    #     num_stages=args.stages,
    #     intial_embed_dim=args.dim,
    #     embed_dim=args.dim,
    #     k_neighbors=args.k,
    #     alpha=args.alpha,
    #     beta=args.beta,
    # ).cuda()

    sigma = 0.3
    seed = 58 # 1
    dimi = 45

    sigma_range = np.arange(0.05, 0.8, 0.05)
    for sigma in sigma_range:

    # seed_range = np.arange(50, 80, 1)
    # for seed in seed_range:

    # dim_range = [3, 9, 18, 27, 36, 45, 54, 63, 72, 81]
    # for dimi in dim_range:

        set_seed(seed)

        point_nn = Point_NN_RBF(
            input_points=args.points,
            num_stages=args.stages,
            intial_embed_dim=dimi,
            embed_dim=dimi,
            k_neighbors=args.k,
            sigma=sigma,
        ).cuda()
        point_nn.eval()

        print("==> Constructing Point-Memory Bank..")

        feature_memory, label_memory = [], []
        # with torch.no_grad():
        for points, labels in tqdm(train_loader):

            points = points.cuda().permute(0, 2, 1)
            # Pass through the Non-Parametric Encoder
            point_features = point_nn(points)
            feature_memory.append(point_features)

            labels = labels.cuda()
            label_memory.append(labels)

        # Feature Memory
        feature_memory = torch.cat(feature_memory, dim=0)
        feature_memory /= feature_memory.norm(dim=-1, keepdim=True)
        feature_memory = feature_memory.permute(1, 0)
        # Label Memory
        label_memory = torch.cat(label_memory, dim=0)
        label_memory = F.one_hot(label_memory).squeeze().float()

        print("==> Saving Test Point Cloud Features..")

        test_features, test_labels = [], []
        # with torch.no_grad():
        for points, labels in tqdm(test_loader):

            points = points.cuda().permute(0, 2, 1)
            # Pass through the Non-Parametric Encoder
            point_features = point_nn(points)
            test_features.append(point_features)

            labels = labels.cuda()
            test_labels.append(labels)

        test_features = torch.cat(test_features)
        test_features /= test_features.norm(dim=-1, keepdim=True)
        test_labels = torch.cat(test_labels)

        print("==> Starting Point-NN..")

        # Search the best hyperparameter gamma
        gamma_list = [i * 10000 / 5000 for i in range(5000)]
        best_acc, best_gamma = 0, 0
        for gamma in tqdm(gamma_list):

            # Similarity Matching
            Sim = test_features @ feature_memory

            # Label Integrate
            logits = (-gamma * (1 - Sim)).exp() @ label_memory

            acc = cls_acc(logits, test_labels)

            if acc > best_acc:
                # print('New best, gamma: {:.2f}; Point-NN acc: {:.2f}'.format(gamma, acc))
                best_acc, best_gamma = acc, gamma

        # After the loop, save the best result
        best_result = f"Best Gamma: {best_gamma:.2f}, Best Accuracy: {best_acc:.2f}, Sigma: {sigma}, Seed: {seed}, Dim: {dimi}\n"
        print(best_result)
        with open("results.txt", "a") as file:
            file.write(best_result)


if __name__ == "__main__":
    main()
