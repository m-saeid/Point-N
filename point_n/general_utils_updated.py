import os
import csv
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.scanobjectnn_dataset import ScanObjectNN
from data.modelnet40_dataset import ModelNet40
from data.modelnet40fewshot_dataset import ModelNet40FewShot


def setup_parser():
    parser = argparse.ArgumentParser(description="Training script for point cloud classification")
    parser.add_argument("--model", type=str, default="pointgn", help="Model type (e.g., pointgn or pointnn)")
    parser.add_argument("--task", type=str, default="cls", help="Task (e.g., classification)")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["scanobject", "modelnet40", "modelnet40fewshot"],
        default="modelnet40",
        help="Dataset to use"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-points", type=int, default=1024, help="Number of points per sample")
    parser.add_argument("--init-dim", type=int, default=72, help="Initial embedding dimension")
    parser.add_argument("--stages", type=int, default=4, help="Number of stages")
    parser.add_argument("--stage-dim", type=int, default=72, help="Dimension after each stage (feature dim)")
    parser.add_argument("--k", type=int, default=90, help="Number of neighbors for kNN")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # New arguments for controlling embedding functions
    parser.add_argument("--embedding_fn", type=str, default="gpe", 
                        help="Embedding function to use: gpe, sine, cosine, tanh, poly, lap, recip, sinc")
    parser.add_argument("--beta", type=int, default=100, help="Beta parameter (if applicable for pointnn)")
    parser.add_argument("--sigma", type=float, default=0.31, help="Sigma value used by embedding functions (e.g., gpe, lap)")
    return parser


def additional_dataset_arguments(parser, dataset):
    if dataset == "modelnet40":
        parser.add_argument("--feat_normalize", action="store_true", default=True, help="Apply feature normalization")
    elif dataset == "modelnet40fewshot":
        parser.add_argument("--n_way", type=int, default=10, choices=[5, 10], help="Number of classes per episode")
        parser.add_argument("--k_shots", type=int, default=20, choices=[10, 20], help="Number of samples per class")
        parser.add_argument("--feat_normalize", action="store_true", default=True, help="Apply feature normalization")
    elif dataset == "scanobject":
        parser.add_argument(
            "--split",
            type=str,
            default="PB_T50_RS",
            choices=["OBJ_BG", "OBJ_ONLY", "PB_T50_RS"],
            help="Split mode for ScanObjectNN"
        )
        parser.add_argument("--feat_normalize", action="store_true", default=False, help="Apply feature normalization")


def additional_model_arguments(parser, model):
    if model == "pointgn":
        # Sigma is now explicitly defined in the base parser.
        pass
    elif model == "pointnn":
        parser.add_argument("--alpha", type=float, default=1000, help="Alpha parameter for pointnn")
        # Beta is already added in the base parser.


def get_arguments():
    parser = setup_parser()
    args, _ = parser.parse_known_args()
    additional_dataset_arguments(parser, args.dataset)
    additional_model_arguments(parser, args.model)
    args = parser.parse_args()
    args.csv_file = generate_csv_filename(args)
    return args


def generate_csv_filename(args):
    csv_file = f"{args.model}_{args.task}_{args.dataset}_n{args.feat_normalize}"
    if args.dataset == "scanobject":
        csv_file += f"_s{args.split}"
    elif args.dataset == "modelnet40fewshot":
        csv_file += f"_n{args.n_way}_k{args.k_shots}"
    return os.path.join("./eval", csv_file + ".csv")


def get_dataloader(dataset, num_points, batch_size, partition, **kwargs):
    num_workers = 8
    return DataLoader(
        dataset(num_points=num_points, partition=partition, **kwargs),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


def get_cls_dataloader(dataset_dir, args):
    if args.dataset == "scanobject":
        dataset_dir = os.path.join(dataset_dir, "scanobjectnn")
        dataset = ScanObjectNN
        kwargs = {"split": args.split}
    elif args.dataset == "modelnet40":
        dataset_dir = os.path.join(dataset_dir, "modelnet")
        dataset = ModelNet40
        kwargs = {}
    elif args.dataset == "modelnet40fewshot":
        dataset_dir = os.path.join(dataset_dir, "modelnet")
        dataset = ModelNet40FewShot
        kwargs = {"n_way": args.n_way, "k_shot": args.k_shots}
    else:
        raise ValueError("Unsupported dataset")

    train_loader = get_dataloader(
        dataset,
        args.num_points,
        args.batch_size,
        "train",
        dataset_dir=dataset_dir,
        **kwargs,
    )
    test_loader = get_dataloader(
        dataset,
        args.num_points,
        args.batch_size,
        "test",
        dataset_dir=dataset_dir,
        **kwargs,
    )

    return train_loader, test_loader


def load_csv_as_dict(csv_file):
    """Loads the CSV file if it exists, or returns an empty list if the file doesn't exist."""
    if os.path.exists(csv_file):
        int_fields = ["batch", "seed", "idim", "fdim", "stage", "k"]
        float_fields = ["sigma"]

        with open(csv_file, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            data = []
            for row in reader:
                for field in int_fields:
                    if field in row and row[field] != "":
                        row[field] = int(row[field])
                for field in float_fields:
                    if field in row and row[field] != "":
                        row[field] = float(row[field])
                data.append(row)
            return data
    else:
        return []


def check_if_combination_exists(data, batch, stage, sigma, seed, init_dim, stage_dim, k):
    """Checks if the given combination has already been calculated in the data."""
    for entry in data:
        if (
            entry["batch"] == batch
            and entry["sigma"] == sigma
            and entry["seed"] == seed
            and entry["idim"] == init_dim
            and entry["fdim"] == stage_dim
            and entry["stage"] == stage
            and entry["k"] == k
        ):
            return True
    return False


def add_new_entry(data, batch, combo, acc_1nn, acc_cos, gamma, train_time, test_time):
    stage, sigma, seed, init_dim, stage_dim, k = combo
    new_entry = {
        "batch": batch,
        "sigma": sigma,
        "seed": seed,
        "idim": init_dim,
        "fdim": stage_dim,
        "stage": stage,
        "k": k,
        "acc_1nn": acc_1nn,
        "acc_cos": acc_cos,
        "gamma": gamma,
        "train_time": train_time,
        "test_time": test_time,
    }
    data.append(new_entry)


def save_data_to_csv(data, filename):
    headers = data[0].keys() if data else []
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_similarity(test_features, train_features, train_labels, test_labels):
    test_features = F.normalize(test_features, dim=-1)
    train_features = F.normalize(train_features, dim=-1)
    gamma_list = [i * 10000 / 5000 for i in range(5000)]
    best_gamma_acc, best_gamma = 0, 0
    for gamma in tqdm(gamma_list, leave=False):
        sim = test_features @ train_features.t()
        logits = torch.exp(-gamma * (1 - sim)) @ train_labels
        pred = logits.topk(1, dim=1, largest=True, sorted=True).indices
        correct = pred.squeeze().eq(test_labels.view(-1))
        acc = correct.float().mean().item() * 100
        if acc > best_gamma_acc:
            best_gamma_acc, best_gamma = acc, gamma
    return best_gamma_acc, best_gamma


def one_nn_classification(test_features, train_features, train_labels, test_labels):
    if train_labels.ndim > 1:
        train_labels = train_labels.argmax(dim=1)
    test_features = F.normalize(test_features, p=2, dim=1)
    train_features = F.normalize(train_features, p=2, dim=1)
    similarity = torch.mm(test_features, train_features.t())
    nearest_indices = similarity.argmax(dim=1)
    pred_labels = train_labels[nearest_indices]
    correct = pred_labels.eq(test_labels)
    accuracy = correct.float().mean().item() * 100.0
    return accuracy
