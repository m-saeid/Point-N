# Non-Parametric Networks for 3D Point Cloud Classification
import math
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

try:
    from ..model_utils import *
except:
    from model_utils import *


# FPS + k-NN
class Local_Grouper(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        # xyz: point coordinates # [B, N, 3]
        # x: point features # [B, N, dim]
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long()
        lc_xyz = index_points(xyz, fps_idx)  # [B, G, 3]
        lc_x = index_points(x, fps_idx)  # [B, G, dim]

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)  # [B, G, K, 3]
        knn_x = index_points(x, knn_idx)  # [B, G, K, feat_dim]

        return lc_xyz, lc_x, knn_xyz, knn_x


# GPE Aggregation
class AggregationGPE(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.geo_extract = PosE_Geo1(3, out_dim, alpha, beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        # lc_xyz =  [B, N, 3]
        # lc_x =    [B, N, D]
        # knn_xyz = [B, N, K, 3]
        # knn_x =   [B, N, K, D]

        # Normalize x (features) and xyz (coordinates)
        center_xyz = lc_xyz.unsqueeze(dim=-2)  # [B, N, 1, 3]
        std_xyz = torch.std(knn_xyz - center_xyz) # [1]

        center_x = lc_x.unsqueeze(dim=-2)  # [B, N, 1, D]
        std_x = torch.std(knn_x - center_x) # [1]

        knn_x = (knn_x - center_x) / (std_x + 1e-5)  # [B, N, K, dim]
        knn_xyz = (knn_xyz - center_xyz) / (std_xyz + 1e-5)  # [B, N, K, 3]

        # Feature Expansion
        b, n, k, d = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(b, n, 1, -1).repeat(1, 1, k, 1)], dim=-1)
        # [B, N, K, D * 2]

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)  # [B, 3, N, K]
        knn_x = knn_x.permute(0, 3, 1, 2)  # [B, D * 2, N, k]
        knn_x_w = self.geo_extract(knn_xyz, knn_x)  # [B, D * 2, N, K]

        return knn_x_w  # [B, D * 2, G, K]


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(nn.BatchNorm1d(out_dim), nn.GELU())

    def forward(self, knn_x_w):
        # knn_x_w = [B, dim, G, K]

        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)  # [B, dim, G]
        lc_x = self.out_transform(lc_x)  # [B, dim, G]
        return lc_x  # [B, dim, G]



class PosE_Initial1(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

        feat_dim = math.ceil(out_dim / (in_dim * 2))
        self.feat_num = feat_dim * 2 * in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long().cuda()

        feat_range = torch.arange(feat_dim).float().cuda()
        self.dim_embed = torch.pow(alpha, feat_range / feat_dim)

    def forward(self, xyz):
        # xyz = [B, in_dim = 3, N]

        b, _, n = xyz.shape  # [B, in_dim = 3, N]

        # Encode x part
        x = xyz[:, 0, :]  # [B, N]
        x_div_embed = torch.div(self.beta * x.unsqueeze(-1), self.dim_embed)  # [B, N, feat_dim]
        x_sin_embed = torch.sin(x_div_embed)  # [B, N, feat_dim]
        x_cos_embed = torch.cos(x_div_embed)  # [B, N, feat_dim]
        x_position_embed = torch.cat([x_sin_embed, x_cos_embed], dim=-1)  # [B, N, feat_dim * 2]
        
        # Encode y part using x_position_embed
        y = xyz[:, 1, :]  # [B, N]
        y_input = torch.cat([y.unsqueeze(-1), x_position_embed], dim=-1)  # [B, N, 1 + feat_dim * 2]
        y_div_embed = torch.div(self.beta * y_input.unsqueeze(-1), self.dim_embed)  # [B, N, 1 + feat_dim * 2, feat_dim]
        y_sin_embed = torch.sin(y_div_embed)  # [B, N, 1 + feat_dim * 2, feat_dim]
        y_cos_embed = torch.cos(y_div_embed)  # [B, N, 1 + feat_dim * 2, feat_dim]
        y_position_embed = torch.cat([y_sin_embed, y_cos_embed], dim=-1)  # [B, N, 1 + feat_dim * 2, feat_dim * 2]
        y_position_embed = y_position_embed.view(b, n, -1)  # [B, N, (1 + feat_dim * 2) * feat_dim * 2]

        # Encode z part using x_position_embed and y_position_embed
        z = xyz[:, 2, :]  # [B, N]
        z_input = torch.cat([z.unsqueeze(-1), x_position_embed, y_position_embed], dim=-1)  # [B, N, 1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2]
        z_div_embed = torch.div(self.beta * z_input.unsqueeze(-1), self.dim_embed)  # [B, N, 1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2, feat_dim]
        z_sin_embed = torch.sin(z_div_embed)  # [B, N, 1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2, feat_dim]
        z_cos_embed = torch.cos(z_div_embed)  # [B, N, 1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2, feat_dim]
        z_position_embed = torch.cat([z_sin_embed, z_cos_embed], dim=-1)  # [B, N, 1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2, feat_dim * 2]
        z_position_embed = z_position_embed.view(b, n, -1)  # [B, N, (1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2) * feat_dim * 2]

        # Combine all embeddings and reshape
        combined_position_embed = torch.cat([x_position_embed, y_position_embed, z_position_embed], dim=-1)  # [B, N, total_dim]
        combined_position_embed = combined_position_embed.permute(0, 2, 1)  # [B, total_dim, N]

        # Select the desired output dimensions
        position_embed = combined_position_embed[:, self.out_idx, :]  # [B, out_dim, N]

        return position_embed  # [B, out_dim, N]




# PosE for Local Geometry Extraction
class PosE_Geo1(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

        feat_dim = math.ceil(out_dim / (in_dim * 2))
        self.feat_num = feat_dim * 2 * in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long().cuda()

        feat_range = torch.arange(feat_dim).float().cuda()
        self.dim_embed = torch.pow(alpha, feat_range / feat_dim)

    def forward(self, knn_xyz, knn_x):
        # knn_xyz = [B, in_dim = 3, N, K]
        # knn_x = [B, out_dim, N, K]

        b, _, n, k = knn_xyz.shape

        # Encode x part
        x = knn_xyz[:, 0, :, :]  # [B, N, K]
        x_div_embed = torch.div(self.beta * x.unsqueeze(-1), self.dim_embed)  # [B, N, K, feat_dim]
        x_sin_embed = torch.sin(x_div_embed)  # [B, N, K, feat_dim]
        x_cos_embed = torch.cos(x_div_embed)  # [B, N, K, feat_dim]
        x_position_embed = torch.cat([x_sin_embed, x_cos_embed], dim=-1)  # [B, N, K, feat_dim * 2]

        # Encode y part using x_position_embed
        y = knn_xyz[:, 1, :, :]  # [B, N, K]
        y_input = torch.cat([y.unsqueeze(-1), x_position_embed], dim=-1)  # [B, N, K, 1 + feat_dim * 2]
        y_div_embed = torch.div(self.beta * y_input.unsqueeze(-1), self.dim_embed)  # [B, N, K, 1 + feat_dim * 2, feat_dim]
        y_sin_embed = torch.sin(y_div_embed)  # [B, N, K, 1 + feat_dim * 2, feat_dim]
        y_cos_embed = torch.cos(y_div_embed)  # [B, N, K, 1 + feat_dim * 2, feat_dim]
        y_position_embed = torch.cat([y_sin_embed, y_cos_embed], dim=-1)  # [B, N, K, 1 + feat_dim * 2, feat_dim * 2]
        y_position_embed = y_position_embed.view(b, n, k, -1)  # [B, N, K, (1 + feat_dim * 2) * feat_dim * 2]

        # Encode z part using x_position_embed and y_position_embed
        z = knn_xyz[:, 2, :, :]  # [B, N, K]
        z_input = torch.cat([z.unsqueeze(-1), x_position_embed, y_position_embed], dim=-1)  # [B, N, K, 1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2]
        z_div_embed = torch.div(self.beta * z_input.unsqueeze(-1), self.dim_embed)  # [B, N, K, 1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2, feat_dim]
        z_sin_embed = torch.sin(z_div_embed)  # [B, N, K, 1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2, feat_dim]
        z_cos_embed = torch.cos(z_div_embed)  # [B, N, K, 1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2, feat_dim]
        z_position_embed = torch.cat([z_sin_embed, z_cos_embed], dim=-1)  # [B, N, K, 1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2, feat_dim * 2]
        z_position_embed = z_position_embed.view(b, n, k, -1)  # [B, N, K, (1 + feat_dim * 2 + (1 + feat_dim * 2) * feat_dim * 2) * feat_dim * 2]

        # Combine all embeddings and reshape
        combined_position_embed = torch.cat([x_position_embed, y_position_embed, z_position_embed], dim=-1)  # [B, N, K, total_dim]
        combined_position_embed = combined_position_embed.permute(0, 3, 1, 2)  # [B, total_dim, N, K]

        # Select the desired output dimensions
        position_embed = combined_position_embed[:, self.out_idx, :, :]  # [B, out_dim, N, K]

        # Weigh
        knn_x_w = knn_x + position_embed  # [B, out_dim, N, K]
        knn_x_w *= position_embed  # [B, out_dim, N, K]

        return knn_x_w  # [B, out_dim, N, K]


# Non-Parametric Encoder
class EncNP(nn.Module):
    def __init__(
        self,
        input_points,
        num_stages,
        intial_embed_dim,
        embed_dim,
        k_neighbors,
        alpha,
        beta,
    ):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta
        self.intial_embed_dim = intial_embed_dim

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(
            3, self.intial_embed_dim, self.alpha, self.beta
        )

        self.Local_Grouper_list = nn.ModuleList()  # FPS, kNN
        self.AggregationGPE_list = nn.ModuleList()  # GPE Aggregation
        self.Pooling_list = nn.ModuleList()  # Pooling

        out_dim = self.intial_embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.Local_Grouper_list.append(Local_Grouper(group_num, k_neighbors))
            self.AggregationGPE_list.append(AggregationGPE(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))

    def forward(self, xyz, x):
        # xyz: point coordinates # [B, N, 3]
        # x: point features # [B, 3, N]

        # Raw-point Embedding
        x = self.raw_point_embed(x)  # [B, intial_embed_dim = 6, 1024]

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.Local_Grouper_list[i](xyz, x.permute(0, 2, 1))
            # xyz =     [B, G_i, 3]
            # lc_x =    [B, G_i, dim_i]
            # knn_xyz = [B, G_i, K, 3]
            # knn_x =   [B, G_i, K, dim_i]

            # GPE Aggregation
            knn_x_w = self.AggregationGPE_list[i](xyz, lc_x, knn_xyz, knn_x)
            # [B, dim_i * 2, G_i, K]

            # Pooling
            x = self.Pooling_list[i](knn_x_w)  # [B, dim_i * 2, G_i]

        # after finish 4 stage loop x  = [B, 96, 64]

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)  # [B, dim = 96]
        return x


# Non-Parametric Network
class Point_NN(nn.Module):
    def __init__(
        self,
        input_points=1024,
        num_stages=4,
        intial_embed_dim=6,
        embed_dim=72,
        k_neighbors=90,
        beta=1000,
        alpha=100,
    ):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncNP(
            input_points,
            num_stages,
            intial_embed_dim,
            embed_dim,
            k_neighbors,
            alpha,
            beta,
        )

    def forward(self, x):
        # xyz: point coordinates
        # x: point features # [B, 3, N]
        xyz = x.permute(0, 2, 1)  # [B, N, 3]

        # Non-Parametric Encoder
        x = self.EncNP(xyz, x)  # [B, dim]
        return x


if __name__ == "__main__":

    sep = 100 * "=" + "\n"

    alpha = 1000
    beta = 100

    in_dim = 3
    embed_dim = 10
    out_dim = embed_dim * 2

    k_neighbors = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_xyz = torch.tensor(
        [[1, 1, 0], [1, 3, 0], [3, 1, 0], [3, 3, 0], [2, 2, 0]],
        device=device,
        dtype=torch.float64,
    )  # [N = 5, in_dim = 3]

    sample_xyz = sample_xyz.unsqueeze(0)  # [B = 1, N = 5, in_dim = 3]
    print(f"{sep}sample_xyz\n{sep}")
    print(sample_xyz[0])
    print(sample_xyz[0].shape)

    pose_init = PosE_Initial(in_dim, embed_dim, alpha, beta)
    sample_x = pose_init.forward(
        sample_xyz.permute(0, 2, 1)
    )  # [B = 1, embed_dim = 10, N = 5]
    sample_x = sample_x.permute(0, 2, 1) # [B = 1, N = 5, embed_dim = 10]
    print(f"{sep}sample_x\n{sep}")
    print(sample_x)
    print(sample_x[0].shape)

    knn_idx = knn_point(k_neighbors, sample_xyz, sample_xyz)  # [B = 1, N = 5, K = 2]
    knn_xyz = index_points(sample_xyz, knn_idx)  # [B = 1, N = 5, K = 2, in_dim = 3]
    knn_x = index_points(sample_x, knn_idx)  # [B = 1, N = 5, K = 2, embed_dim = 10]
    print(f"{sep}knn_xyz\n{sep}")
    print(knn_xyz)
    print(knn_xyz[0].shape)

    AggregationGPE = AggregationGPE(out_dim, alpha, beta)


    new_knn_x = AggregationGPE.forward(sample_xyz, sample_x, knn_xyz, knn_x)
    # [B = 1, out_dim = 20, N = 5, K = 2]
    new_knn_x = new_knn_x.permute(0, 2, 3, 1) # [B = 1, N = 5, K = 2, out_dim = 20]
    print(f"{sep}new_knn_x\n{sep}")
    print(new_knn_x)
    print(new_knn_x[0].shape)
