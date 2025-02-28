# Non-Parametric Networks for 3D Point Cloud Part Segmentation
import math
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils


try:
    from ..model_utils import index_points, knn_point
except:
    from model_utils import index_points, knn_point



#Local Grouper : FPS + k-NN + Normalization
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
        fps_idx = pointnet2_utils.furthest_point_sample(
            xyz.contiguous(), self.group_num
        ).long()
        lc_xyz = index_points(xyz, fps_idx)  # [B, G, 3]
        lc_x = index_points(x, fps_idx)  # [B, G, dim]

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)  # [B, G, K, 3]
        knn_x = index_points(x, knn_idx)  # [B, G, K, feat_dim]

        # Normalize x (features) and xyz (coordinates)
        center_xyz = lc_xyz.unsqueeze(dim=-2)  # [B, G, 1, 3]
        std_xyz = torch.std(knn_xyz - center_xyz)  # [1]
        knn_xyz = (knn_xyz - center_xyz) / (std_xyz + 1e-5)  # [B, G, K, 3]

        center_x = lc_x.unsqueeze(dim=-2)  # [B, G, 1, D]
        std_x = torch.std(knn_x - center_x)  # [1]
        knn_x = (knn_x - center_x) / (std_x + 1e-5)  # [B, G, K, D]

        # Feature Expansion
        b, g, k, _ = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(b, g, 1, -1).repeat(1, 1, k, 1)], dim=-1)

        return lc_xyz, lc_x, knn_xyz, knn_x


# GPE Aggregation
class AggregationGPE(nn.Module):
    def __init__(self, out_dim, sigma):
        super().__init__()
        
        self.geo_extract = EmbeddingGPE(3, out_dim, sigma)

    def forward(self,knn_xyz, knn_x):
        # knn_xyz = [B, G, K, 3]
        # knn_x =   [B, G, K, dim]

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)  # [B, 3, G, K]
        knn_x = knn_x.permute(0, 3, 1, 2)  # [B, D*2, G, k]

        # Weigh
        position_embed = self.geo_extract(knn_xyz)  # [B, D*2, G, k]
        knn_x_w = knn_x + position_embed  # [B, D*2, G, K]
        knn_x_w *= position_embed  # [B, D*2, G, K]

        return knn_x_w  # [B, D * 2, G, K]


class EmbeddingGPE(nn.Module):
    def __init__(self, in_dim, out_dim, sigma):
        super(EmbeddingGPE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma

        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * self.in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long().cuda()

        self.feat_val = (
            torch.linspace(-1.0, 1.0, feat_dim + 1)[:-1].reshape(1, -1).cuda()
        )

    def forward(self, xyz):
        # xyz = [B, in_dim = 3, N] or [B, in_dim = 3, G, K]

        if xyz.dim() not in {3, 4}:
            raise ValueError("Input must be either [B, in_dim, N] or [B, in_dim, G, K]")

        xyz_p = xyz.permute(0, *range(2, xyz.ndim), 1)
        # xyz_p = [B, ..., 3]

        # Initialize a list to store the embeddings for each channel
        embeds = []

        # Compute the RBF features for each channel in a loop
        for i in range(3):
            embed = (
                -0.5 * (xyz_p[..., i : i + 1] - self.feat_val) ** 2 / (self.sigma**2)
            ).exp()
            embeds.append(embed)

        # Concatenate the embeddings along the last dimension
        position_embed = torch.cat(embeds, dim=-1)
        # [B, ...,  feat_dim * 2]

        # Reshape and permute position embedding based on input dimensions
        if xyz.dim() == 3:
            b, _, n = xyz.shape
            position_embed = position_embed.permute(0, 2, 1).reshape(
                b, self.feat_num, n
            )  # [B, feat_num, N]
        elif xyz.dim() == 4:
            b, _, g, k = xyz.shape
            position_embed = position_embed.permute(0, 3, 1, 2).reshape(
                b, self.feat_num, g, k
            )  # [B, feat_num, G, K]

        return position_embed  # [B, out_dim, ...]


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(nn.BatchNorm1d(out_dim), nn.GELU())

    def forward(self, knn_x_w):
        # knn_x_w = [B, dim, G, K]

        # Feature Aggregation (Pooling)
        agg_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)  # [B, dim, G]
        agg_x = self.out_transform(agg_x)  # [B, dim, G]
        return agg_x  # [B, dim, G]

# Non-Parametric Encoder
class EncNP(nn.Module):  
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, sigma):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.sigma = sigma

        # Raw-point Embedding
        self.raw_point_embed = EmbeddingGPE(3, self.embed_dim, sigma)

        self.Local_Grouper_list = nn.ModuleList() # FPS, kNN
        self.AggregationGPE_list = nn.ModuleList() # GPE Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.Local_Grouper_list.append(Local_Grouper(group_num, k_neighbors))
            self.AggregationGPE_list.append(AggregationGPE(out_dim, self.sigma))
            self.Pooling_list.append(Pooling(out_dim))


    def forward(self, xyz, x):

        # Raw-point Embedding
        x = self.raw_point_embed(x)

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, C, N]

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.Local_Grouper_list[i](xyz, x.permute(0, 2, 1))
            # GPE Aggregation
            knn_x_w = self.AggregationGPE_list[i]( knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

            xyz_list.append(xyz)
            x_list.append(x)

        return xyz_list, x_list


# Non-Parametric Decoder
class DecNP(nn.Module):  
    def __init__(self, num_stages, de_neighbors):
        super().__init__()
        self.num_stages = num_stages
        self.de_neighbors = de_neighbors


    def propagate(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            weight = weight.view(B, N, self.de_neighbors, 1)

            index_points(xyz1, idx)
            interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)

        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        return new_points


    def forward(self, xyz_list, x_list):
        xyz_list.reverse()
        x_list.reverse()

        x = x_list[0]
        for i in range(self.num_stages):
            # Propagate point features to neighbors
            x = self.propagate(xyz_list[i+1], xyz_list[i], x_list[i+1], x)
        return x


# Non-Parametric Network
class Point_NN_Seg(nn.Module):
    def __init__(self, input_points=2048, num_stages=5, embed_dim=144, 
                    k_neighbors=128, de_neighbors=6, sigma=0.3):
        super().__init__()
        # Non-Parametric Encoder and Decoder
        self.EncNP = EncNP(input_points, num_stages, embed_dim, k_neighbors,sigma)
        self.DecNP = DecNP(num_stages, de_neighbors)


    def forward(self, x):
        # xyz: point coordinates
        # x: point features
        xyz = x.permute(0, 2, 1)

        # Non-Parametric Encoder
        xyz_list, x_list = self.EncNP(xyz, x)

        # Non-Parametric Decoder
        x = self.DecNP(xyz_list, x_list)
        return x
