import os
import time
import itertools

from tqdm import tqdm

import torch
import torch.nn.functional as F

import general_utils as gutils
from models.point_gn import PointGNCls


def generate_parameter_combinations(args):
    if args.dataset == "modelnet40":

        sigma_range = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        # [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        seed_range = [42]
        init_dim_range = [9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117]
        stage_range = [2, 3, 4]  # [3, 4]  # [3, 4, 5, 6, 7]
        stage_dim_range = [18, 27, 36, 45, 54, 63, 72, 81, 90, 99]
        # [36, 54, 72, 90, 99, 117]
        k_range = [70, 80, 90, 100, 110, 120, 130]
        # [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

    elif args.dataset == "scanobject":
        sigma_range = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        # [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        seed_range = [42]
        init_dim_range = [9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117]
        stage_range = [2, 3, 4]  # [3, 4]  # [3, 4, 5, 6, 7]
        stage_dim_range = [18, 27, 36, 45, 54, 63, 72, 81, 90, 99]
        # [36, 54, 72, 90, 99, 117]
        k_range = [70, 80, 90, 100, 110, 120, 130]
        # [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
        
        # sigma_range = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        # # [0.2, 0.3, 0.4, 0.5]
        # seed_range = [42]
        # init_dim_range = [9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117]
        # stage_range = [3, 4]  # [3, 4]  # [3, 4, 5, 6, 7]
        # stage_dim_range = [27, 36, 45, 54, 63, 72, 81, 90, 99]
        # # [36, 54, 72, 90, 99, 117]
        # k_range = [80, 90, 100, 110, 120, 130]
        # # [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

    # Generate all combinations of parameters
    combinations = itertools.product(
        stage_range, sigma_range, seed_range, init_dim_range, stage_dim_range, k_range
    )

    return [list(combination) for combination in combinations]


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


@torch.no_grad()
def main_cls_param():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    dataset_dir = os.path.join(project_root, "datasets")

    args = gutils.get_arguments()
    if args.model == "pointnn":
        raise ValueError("not correct")
    train_loader, test_loader = gutils.get_cls_dataloader(dataset_dir, args)

    saved_data = gutils.load_csv_as_dict(args.csv_file)
    combinations = generate_parameter_combinations(args)
    for sublist in combinations:
        sublist[3] = sublist[4]

    unique_combinations = set()

    for combo in combinations:
        combo_tuple = tuple(combo)
        if not gutils.check_if_combination_exists(saved_data, args.batch_size, *combo):
            if combo_tuple not in unique_combinations:
                unique_combinations.add(combo_tuple)

    for combo in tqdm(unique_combinations):
        stage, sigma, seed, init_dim, stage_dim, k = combo

        if k >= args.num_points // (2 ** (stage - 1)):
            print("k is out of range")
            continue

        gutils.set_seed(seed)

        model = PointGNCls(
            num_points=args.num_points,
            init_dim=init_dim,
            stages=stage,
            stage_dim=stage_dim,
            k=k,
            sigma=sigma,
            feat_normalize=args.feat_normalize,
        )
        model.to(device).eval()

        start_train_time = time.time()
        train_features, train_labels = process_data(train_loader, model, device)
        # train_features: [num_features, num_train_samples]
        # train_labels :[num_train_samples, 1]
        train_labels = F.one_hot(train_labels).squeeze().float()
        # [num_train_samples, num_classes]
        train_time = time.time() - start_train_time

        start_test_time = time.time()
        test_features, test_labels = process_data(test_loader, model, device)
        # test_features: [num_features, num_test_samples]
        # test_labels :[num_test_samples, 1]
        test_time = time.time() - start_test_time

        acc_cos, gamma = gutils.cosine_similarity(
            test_features, train_features, train_labels, test_labels
        )

        acc_1nn = gutils.one_nn_classification(
            test_features, train_features, train_labels, test_labels
        )

        gutils.add_new_entry(
            saved_data,
            args.batch_size,
            combo,
            acc_1nn,
            acc_cos,
            gamma,
            train_time,
            test_time,
        )
        gutils.save_data_to_csv(saved_data, args.csv_file)
        # print(
        #     f"Saved: sigma={sigma}, seed={seed}, idim={init_dim}, fdim={stage_dim}, stage={stage}, k={k}, acc={best_accuracy:.4f}, gamma={best_gamma:.4f}"
        # )


if __name__ == "__main__":
    main_cls_param()
