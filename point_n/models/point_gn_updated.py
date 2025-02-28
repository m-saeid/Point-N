import os
import sys
import math
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

# Path setup
project_path = os.path.abspath(".")
sys.path.append(project_path)

from models.model_utils import index_points, knn_point, farthest_point_sample

def normalize_tensor(data_tensor, knn_tensor, with_center=True):
    if with_center:
        center_tensor = data_tensor.unsqueeze(dim=-2)
        knn_std = torch.std(knn_tensor - center_tensor, dim=(0, 1, 3), keepdim=True).clamp(min=1e-5)
        normalized_knn_tensor = (knn_tensor - center_tensor) / knn_std
    else:
        knn_std = torch.std(knn_tensor, dim=(0, 1), keepdim=True).clamp(min=1e-5)
        normalized_knn_tensor = knn_tensor / knn_std
    return normalized_knn_tensor

class LocalGrouper(nn.Module):
    def __init__(self, stage_points, k, feat_normalize):
        super().__init__()
        self.stage_points = stage_points
        self.k = k
        self.feat_normalize = feat_normalize

    def forward(self, xyz, feat):
        b, n, _ = xyz.shape
        if xyz.device == torch.device("cpu"):
            fps_idx = farthest_point_sample(xyz.contiguous(), self.stage_points).long()
        else:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.stage_points).long()
        xyz_sampled = index_points(xyz, fps_idx)
        feat_sampled = index_points(feat, fps_idx)
        idx_knn = knn_point(self.k, xyz, xyz_sampled)
        xyz_knn = index_points(xyz, idx_knn)
        feat_knn = index_points(feat, idx_knn)
        xyz_knn = normalize_tensor(xyz_sampled, xyz_knn, with_center=True)
        feat_knn = normalize_tensor(feat_sampled, feat_knn, with_center=self.feat_normalize)
        b, s, k, _ = feat_knn.shape
        feat_knn = torch.cat([feat_knn, feat_sampled.reshape(b, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)
        return xyz_sampled, feat_sampled, xyz_knn, feat_knn

# (Existing classes: AggregationGPE, EmbeddingGPE, EmbeddingSine, etc.)

class EmbeddingGPE(nn.Module):
    def __init__(self, in_dim, out_dim, sigma):
        super(EmbeddingGPE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * self.in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)

    def forward(self, xyz):
        if self.out_dim == 0:
            return xyz
        if xyz.dim() not in {3, 4}:
            raise ValueError("Input must be either [B, N, 3] or [B, S, K, 3]")
        embeds = []
        for i in range(self.in_dim):
            tmp = xyz[..., i:i+1] - self.feat_val.to(xyz.device)
            embed = (-0.5 * tmp**2 / (self.sigma**2)).exp()
            embeds.append(embed)
        position_embed = torch.cat(embeds, dim=-1)
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))
        return position_embed

# New Hybrid/Augmented Embedding Function:
class EmbeddingHybridAugmented(nn.Module):
    """
    Hybrid/Augmented Embedding Function
    
    Assumptions:
    1. Complementarity: Combines the RBF (Gaussian) embedding and a complementary cosine embedding.
       - RBF captures local Euclidean similarity.
       - Cosine highlights periodic/angle-related aspects.
    2. Parameterlessness: Fusion is achieved with fixed weights (a blend factor) without adding learnable parameters.
    3. Fusion Strategy: Both embeddings are computed to produce vectors of the same size.
       An element-wise weighted sum is then used to fuse them.
    4. Robustness: The combined representation is assumed to be richer, capturing more nuances.
    5. No Extra Learning Overhead: All operations are fixed functions.
    
    For each input channel, computes:
      - RBF component: exp(-0.5 * ((x - v)/sigma)^2)
      - Cosine component: cos(x - v)
    Fuses them as: output = blend * RBF + (1 - blend) * cosine.
    """
    def __init__(self, in_dim, out_dim, sigma, blend=0.5):
        super(EmbeddingHybridAugmented, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.blend = blend  # Fixed blending weight
        
        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)
    
    def forward(self, xyz):
        if self.out_dim == 0:
            return xyz
        if xyz.dim() not in {3, 4}:
            raise ValueError("Input must be either [B, N, in_dim] or [B, S, K, in_dim]")
        embeds = []
        for i in range(self.in_dim):
            tmp = xyz[..., i:i+1] - self.feat_val.to(xyz.device)
            # Compute RBF component (Gaussian)
            rbf = (-0.5 * tmp**2 / (self.sigma**2)).exp()
            # Compute complementary cosine component
            cosine = torch.cos(tmp)
            # Fuse them using fixed blending weights
            combined = self.blend * rbf + (1 - self.blend) * cosine
            embeds.append(combined)
        position_embed = torch.cat(embeds, dim=-1)
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))
        return position_embed


class EmbeddingGaussianLaplacianFusion(nn.Module):
    """
    A parameterless fusion embedding function that combines the Gaussian (RBF) and
    Laplacian embeddings. The Gaussian component captures smooth local similarity,
    while the Laplacian component is more sensitive to sharp differences.
    
    The fusion is performed using a fixed blending weight (blend) so that:
    
        output = blend * Gaussian + (1 - blend) * Laplacian
    
    Assumptions:
      1. Complementarity: Gaussian and Laplacian kernels capture slightly different
         aspects of local geometry.
      2. Parameterlessness: Fusion is done with fixed weights (e.g., blend = 0.5).
      3. Fusion Strategy: Both embeddings are computed over the same grid and then
         fused element-wise.
      4. Robustness: The combined representation may capture a richer set of features.
    """
    def __init__(self, in_dim, out_dim, sigma, blend=0.5):
        super(EmbeddingGaussianLaplacianFusion, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.blend = blend  # fixed blending weight
        
        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * in_dim
        # Create an index to select the first 'out_dim' features after concatenation.
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        # Create a fixed grid of feature values between -1 and 1 (excluding endpoints).
        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)
    
    def forward(self, xyz):
        """
        Args:
            xyz: Input tensor of shape [B, N, in_dim] or [B, S, K, in_dim]
        
        Returns:
            A tensor of shape [B, ..., out_dim] that fuses the Gaussian and Laplacian embeddings.
        """
        if self.out_dim == 0:
            return xyz
        if xyz.dim() not in {3, 4}:
            raise ValueError("Input must be either [B, N, in_dim] or [B, S, K, in_dim]")
        
        gaussian_embeds = []
        laplacian_embeds = []
        for i in range(self.in_dim):
            # Compute the difference from fixed values for the i-th coordinate.
            tmp = xyz[..., i:i+1] - self.feat_val.to(xyz.device)
            # Gaussian (RBF) component: exp(-0.5 * (tmp/sigma)^2)
            gaussian = (-0.5 * tmp**2 / (self.sigma**2)).exp()
            gaussian_embeds.append(gaussian)
            # Laplacian component: exp(-|tmp|/sigma)
            laplacian = torch.exp(-torch.abs(tmp) / self.sigma)
            laplacian_embeds.append(laplacian)
        
        # Concatenate embeddings along the last dimension.
        gaussian_cat = torch.cat(gaussian_embeds, dim=-1)
        laplacian_cat = torch.cat(laplacian_embeds, dim=-1)
        # Fuse the two components with a fixed weight.
        fused = self.blend * gaussian_cat + (1 - self.blend) * laplacian_cat
        # Select the required output dimensions.
        fused = torch.index_select(fused, -1, self.out_idx.to(xyz.device))
        return fused








class EncoderGPECls(nn.Module):
    def __init__(self, num_points, init_dim, stages, stage_dim, k, sigma, feat_normalize, embedding_fn="gpe"):
        super().__init__()
        self.num_points = num_points
        self.init_dim = init_dim
        self.stages = stages
        self.stage_dim = stage_dim
        self.sigma = sigma

        # Update the embedding mapping to include the new hybrid function.
        embedding_map = {
            "gpe": EmbeddingGPE,
            "sine": EmbeddingSine,
            "cosine": EmbeddingCosine,
            "tanh": EmbeddingTanh,
            "poly": EmbeddingPolynomial,
            "lap": EmbeddingLaplacian,
            "recip": EmbeddingReciprocal,
            "sinc": EmbeddingSinc,
            "hybrid": EmbeddingHybridAugmented  # New hybrid function
            "gausslap": EmbeddingGaussianLaplacianFusion

        }
        self.embed_cls = embedding_map.get(embedding_fn, EmbeddingGPE)
        self.init_embed = self.embed_cls(3, self.init_dim, self.sigma)

        self.lg_list = nn.ModuleList()
        self.agpe_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()

        out_dim = self.init_dim if self.init_dim != 0 else 3
        stage_points = self.num_points

        for i in range(self.stages):
            out_dim *= 2
            stage_points //= 2
            self.lg_list.append(LocalGrouper(stage_points, k, feat_normalize))
            self.agpe_list.append(AggregationGPE(out_dim, self.sigma, embedding_fn_cls=self.embed_cls))
            self.pool_list.append(NeighborPooling(out_dim))

    def forward(self, xyz):
        feat = self.init_embed(xyz)
        stage_results = []
        for i in range(self.stages):
            xyz, feat, xyz_knn, feat_knn = self.lg_list[i](xyz, feat)
            feat_knn_w = self.agpe_list[i](xyz_knn, feat_knn)
            feat = self.pool_list[i](feat_knn_w)
            stage_pooling = torch.cat((feat.max(-2)[0], feat.mean(-2)), dim=1)
            stage_results.append(stage_pooling)
        encoded_out = torch.cat(stage_results, dim=1)
        return encoded_out

class PointGNCls(nn.Module):
    def __init__(self, num_points=1024, init_dim=6, stages=4, stage_dim=72, k=90, sigma=0.3, feat_normalize=True, embedding_fn="gpe"):
        super().__init__()
        self.EncNP = EncoderGPECls(num_points, init_dim, stages, stage_dim, k, sigma, feat_normalize, embedding_fn)

    def forward(self, xyz):
        x = self.EncNP(xyz)
        return x

if __name__ == "__main__":
    batch_size = 128
    num_points = 1024
    in_ch = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    random_xyz = torch.randn(batch_size, num_points, in_ch).contiguous().to(device)
    model = PointGNCls(num_points=num_points).to(device)
    start_time = time.time()
    output = model(random_xyz)
    end_time = time.time()
    print("Output shape: {}".format(output.shape))
    print("Time taken for forward pass: {:.6f} seconds".format(end_time - start_time))
