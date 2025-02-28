import os
import sys
import math
import torch
import torch.nn as nn

# from pointnet2_ops import pointnet2_utils #################################################################################

project_path = os.path.abspath(".")
sys.path.append(project_path)

from models.model_utils import index_points, knn_point, farthest_point_sample


def compute_neighbors(xyz, k=16):
    """
    Computes the k-nearest neighbors for a given point cloud.
    
    Args:
        xyz (torch.Tensor): Tensor of shape (B, N, 3), representing the point cloud.
        k (int): Number of neighbors to find.
    
    Returns:
        neighbor_xyz (torch.Tensor): Tensor of shape (B, N, K, 3), representing the k-nearest neighbors.
    """
    B, N, _ = xyz.shape

    # Compute pairwise distances between all points
    dist_matrix = torch.cdist(xyz, xyz)  # Shape: (B, N, N)

    # Find the k smallest distances (excluding self, index 0 is itself)
    knn_idx = torch.topk(dist_matrix, k=k+1, largest=False, dim=-1).indices[:, :, 1:]  # Shape: (B, N, K)

    # Gather the k-nearest neighbor points
    neighbor_xyz = torch.gather(
        xyz.unsqueeze(2).expand(-1, -1, k, -1),  # Shape: (B, N, K, 3)
        dim=1,
        index=knn_idx.unsqueeze(-1).expand(-1, -1, -1, xyz.shape[-1])
    )

    return neighbor_xyz  # Shape: (B, N, K, 3)



def adaptive_normalization(knn_tensor, reference_tensor, mode="local"):
    """
    Adaptive normalization function.
    
    Args:
        knn_tensor: Neighborhood tensor of shape [B, N, K, in_dim].
        reference_tensor: Reference point tensor of shape [B, N, in_dim].
        mode: "global" for traditional normalization, "local" for per-neighborhood normalization.
    
    Returns:
        Normalized tensor of the same shape as knn_tensor.
    """
    if mode == "global":
        std = torch.std(knn_tensor, dim=(0, 1), keepdim=True).clamp(min=1e-5)
    elif mode == "local":
        std = torch.std(knn_tensor - reference_tensor.unsqueeze(dim=2), dim=2, keepdim=True).clamp(min=1e-5)
    else:
        raise ValueError("Unknown normalization mode.")
    return knn_tensor / std

def compute_local_mad(neighbor_xyz, reference_xyz):
    """
    Compute the Median Absolute Deviation (MAD) for each neighborhood.
    :param neighbor_xyz: Tensor of shape [B, N, K, in_dim]
    :param reference_xyz: Tensor of shape [B, N, in_dim]
    :return: Tensor of MAD values per point.
    """
    local_distances = torch.norm(neighbor_xyz - reference_xyz.unsqueeze(2), dim=-1)
    median = local_distances.median(dim=-1, keepdim=True).values
    mad = (local_distances - median).abs().median(dim=-1, keepdim=True).values
    return mad * 1.4826  # Correction factor to approximate standard deviation


def compute_local_curvature(neighbor_xyz):
    """
    Compute curvature from local neighborhood points.
    
    For 4D input (e.g., [B, N, K, D]): 
      - Performs PCA on each neighborhood and averages the curvature over the K dimension.
      - Returns a tensor of shape [B, N, 1].
    
    For 3D input (e.g., [B, K, D]):
      - Computes curvature for each group (using PCA) and returns a tensor of shape [B, 1].
    
    This updated function is backward-compatible for modules (like EmbeddingMultiHybrid_2)
    that expect 4D inputs, while also supporting 3D inputs for modules like EnrichedGeometricEmbedding.
    """
    if neighbor_xyz.dim() == 4:
        # 4D case: expected for EmbeddingMultiHybrid_2
        B, N, K, D = neighbor_xyz.shape
        centered = neighbor_xyz - neighbor_xyz.mean(dim=2, keepdim=True)
        # Sum over the neighbor dimension to get a square covariance matrix of shape [B, N, D, D]
        covariance = torch.einsum("bnki,bnkj->bnij", centered, centered) / (K - 1)
        eigenvalues, _ = torch.linalg.eigh(covariance)
        # Curvature computed as the ratio of the smallest eigenvalue to the sum of eigenvalues
        curvature = eigenvalues[..., 0] / (eigenvalues.sum(dim=-1) + 1e-6)  # shape: [B, N]
        curvature = curvature.unsqueeze(-1)  # shape: [B, N, 1]
        return curvature
    elif neighbor_xyz.dim() == 3:
        # 3D case: expected for EnrichedGeometricEmbedding (after reshaping)
        B, K, D = neighbor_xyz.shape
        centered = neighbor_xyz - neighbor_xyz.mean(dim=1, keepdim=True)
        covariance = torch.matmul(centered.transpose(1, 2), centered) / (K - 1)
        eigenvalues, _ = torch.linalg.eigh(covariance)
        min_eig = eigenvalues[:, 0].unsqueeze(-1)
        sum_eig = eigenvalues.sum(dim=1, keepdim=True)
        curvature = min_eig / (sum_eig + 1e-6)
        return curvature
    else:
        raise ValueError("Unsupported input dimension for compute_local_curvature")




def normalize_tensor(data_tensor, knn_tensor, with_center=True):
    if with_center:
        center_tensor = data_tensor.unsqueeze(dim=-2)
        knn_std = torch.std(knn_tensor - center_tensor, dim=(0, 1, 3), keepdim=True).clamp(min=1e-5)
        normalized_knn_tensor = (knn_tensor - center_tensor) / knn_std
    else:
        knn_std = torch.std(knn_tensor, dim=(0, 1), keepdim=True).clamp(min=1e-5)
        normalized_knn_tensor = knn_tensor / knn_std
    return normalized_knn_tensor



def compute_local_sigma_per_point(neighbor_xyz, reference_xyz):
    """
    Compute adaptive sigma per point using standard deviation of distances in a local neighborhood.
    :param neighbor_xyz: Tensor of shape [B, N, K, in_dim]
    :param reference_xyz: Tensor of shape [B, N, in_dim]
    :return: Tensor of shape [B, N] containing the adaptive sigma for each point.
    """
    local_std = torch.std(neighbor_xyz - reference_xyz.unsqueeze(2), dim=2, keepdim=True)
    return local_std.mean(dim=-1)

def compute_blend_weights(neighbor_xyz, reference_xyz, mode="curvature"):
    """
    Compute blending weights based on local curvature or density.
    :param neighbor_xyz: Tensor of shape [B, N, K, in_dim]
    :param reference_xyz: Tensor of shape [B, N, in_dim]
    :param mode: 'curvature' or 'density'
    :return: Tensor of blend weights with shape matching the embedding outputs.
    """
    if mode == "curvature":
        curvatures = compute_local_curvature(neighbor_xyz)
        blend_weights = torch.sigmoid(10 * (curvatures - curvatures.mean(dim=1, keepdim=True)))
    elif mode == "density":
        density = torch.mean(torch.norm(neighbor_xyz - reference_xyz.unsqueeze(2), dim=-1, keepdim=True), dim=2)
        blend_weights = torch.sigmoid(10 * (density - density.mean(dim=1, keepdim=True)))
    return blend_weights



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
            # fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.stage_points).long() #########################
            fps_idx = farthest_point_sample(xyz.contiguous(), self.stage_points).long() 
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

class AggregationGPE(nn.Module):
    def __init__(self, out_dim, sigma, embedding_fn_cls):
        super().__init__()
        self.gpe_embed = embedding_fn_cls(3, out_dim, sigma)

    def forward(self, xyz_knn, feat_knn):
        # If the embedding function requires neighbor_xyz (i.e. if its forward method takes 2 arguments),
        # pass xyz_knn as both the input and the neighbor information.
        if isinstance(self.gpe_embed, (EmbeddingMultiHybrid_2, EnrichedGeometricEmbedding)):
            position_embed = self.gpe_embed(xyz_knn, xyz_knn)
        else:
            position_embed = self.gpe_embed(xyz_knn)
        feat_knn_w = feat_knn + position_embed
        feat_knn_w *= position_embed
        return feat_knn_w


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
            embed = -0.5 * tmp**2 / (self.sigma**2)
            embeds.append(embed.exp())
        position_embed = torch.cat(embeds, dim=-1)
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))
        return position_embed


class AdaptiveEmbeddingGPE_1(nn.Module):
    """
    Adaptive Embedding Function (AdaptiveEmbeddingGPE)
    
    This function implements an adaptive, data-driven variant of the standard
    Gaussian (RBF) embedding. It adjusts the kernel width (sigma) based on the 
    global standard deviation of the input and uses an adaptive blending strategy
    to fuse the Gaussian response with a complementary cosine response.
    
    Assumptions and Implementation Details:
    
    1. Adaptive Kernel Width:
       - Compute a global standard deviation from the input (over points) and 
         adjust the effective sigma as: adaptive_sigma = base_sigma * (1 + global_std).
         
    2. Adaptive Blending:
       - Compute a blend weight (between 0 and 1) as: 
           blend = sigmoid((global_std - baseline) * scaling)
         This weight is used to fuse the Gaussian (RBF) embedding with a cosine embedding.
         
    3. Dynamic Normalization:
       - The difference (tmp) is divided by the adaptive sigma to normalize the scale 
         of the kernel function.
    
    4. Complementarity:
       - The Gaussian captures local similarity via an exponential decay,
         whereas the cosine transformation introduces a periodic component.
         Their fusion is intended to yield a richer representation.
    
    5. Parameterlessness:
       - All adaptation is computed on-the-fly from the data, with no learnable parameters.
    
    Args:
      in_dim (int): Input dimension (typically 3 for XYZ coordinates).
      out_dim (int): Desired output dimension.
      sigma (float): Base sigma value (default kernel width).
      baseline (float): A fixed baseline for computing blend weight (default 0.1).
      scaling (float): Scaling factor for the sigmoid to compute blend (default 10.0).
      eps (float): Small constant to prevent division by zero.
    """
    def __init__(self, in_dim, out_dim, sigma, baseline=0.1, scaling=10.0, eps=1e-6):
        super(AdaptiveEmbeddingGPE_1, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_sigma = sigma  # base kernel width
        self.baseline = baseline
        self.scaling = scaling
        self.eps = eps
        
        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        # Fixed grid of values for embedding (excluding endpoints)
        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)
        
    def forward(self, xyz):
        """
        Args:
          xyz: Tensor of shape [B, N, in_dim] or [B, S, K, in_dim]
        Returns:
          Tensor of shape [B, ..., out_dim] computed by adaptively fusing a Gaussian 
          and a cosine response.
        """
        if self.out_dim == 0:
            return xyz
        if xyz.dim() == 3:
            # Compute global standard deviation across points (dim=1)
            global_std = torch.mean(torch.std(xyz, dim=1))
        elif xyz.dim() == 4:
            # Reshape to [B, -1, in_dim] and compute standard deviation over points
            global_std = torch.mean(torch.std(xyz.view(xyz.size(0), -1, self.in_dim), dim=1))
        else:
            raise ValueError("Input must be 3D or 4D")
        
        # Adaptive sigma: scale the base sigma by (1 + global_std)
        adaptive_sigma = self.base_sigma * (1 + global_std)
        # Adaptive blend weight via sigmoid; yields a value in (0,1)
        blend = torch.sigmoid((global_std - self.baseline) * self.scaling)
        
        embeds = []
        for i in range(self.in_dim):
            # Compute difference from fixed grid values
            tmp = xyz[..., i:i+1] - self.feat_val.to(xyz.device)
            # Gaussian (RBF) component using adaptive sigma
            rbf = (-0.5 * (tmp / (adaptive_sigma + self.eps))**2).exp()
            # Cosine component using the same adaptive sigma for scaling
            cosine = torch.cos(tmp / (adaptive_sigma + self.eps))
            # Adaptive fusion of the two components:
            combined = blend * rbf + (1 - blend) * cosine
            embeds.append(combined)
        
        # Concatenate all channels and select the desired output dimensions
        position_embed = torch.cat(embeds, dim=-1)
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))
        return position_embed

class AdaptiveEmbeddingGPE_2(nn.Module):
    """
    Adaptive Gaussian Positional Encoding (Adaptive GPE)
    
    - Dynamically adjusts sigma based on local variance or density.
    - Computes adaptive blending weights using local geometric properties.
    """
    def __init__(self, in_dim, out_dim, base_sigma=0.3, blend_strategy="variance", eps=1e-6):
        super(AdaptiveEmbeddingGPE_2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_sigma = base_sigma
        self.blend_strategy = blend_strategy
        self.eps = eps

        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)

    def forward(self, xyz, neighbor_xyz):
        """
        Args:
            xyz: Input point cloud tensor of shape [B, N, in_dim].
            neighbor_xyz: Neighborhood points tensor of shape [B, N, K, in_dim].
        Returns:
            Adaptive positional encoding of shape [B, N, out_dim].
        """
        if self.out_dim == 0:
            return xyz

        # Compute local standard deviation or mean distance
        if self.blend_strategy == "variance":
            local_stat = torch.std(neighbor_xyz - xyz.unsqueeze(2), dim=2, keepdim=True)
        elif self.blend_strategy == "density":
            local_stat = torch.mean(torch.norm(neighbor_xyz - xyz.unsqueeze(2), dim=-1, keepdim=True), dim=2)
        else:
            raise ValueError("Unknown blend strategy.")

        # Compute adaptive sigma
        adaptive_sigma = self.base_sigma * (1 + local_stat)

        # Compute Gaussian embedding
        embeds = []
        for i in range(self.in_dim):
            # Expand self.feat_val to [B, N, feat_dim]
            feat_val_expanded = self.feat_val.to(xyz.device).expand(xyz.size(0), xyz.size(1), -1)
            tmp = xyz[..., i:i+1] - feat_val_expanded  # Now tmp will have shape [B, N, feat_dim]
            sigma_i = adaptive_sigma[..., i:i+1].squeeze(2)  # shape: [B, N, 1]
            rbf = (-0.5 * (tmp / (sigma_i + self.eps))**2).exp()
            embeds.append(rbf)



        position_embed = torch.cat(embeds, dim=-1)
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))
        return position_embed


class EmbeddingMultiHybrid_3(nn.Module):
    """
    Multi-Hybrid Embedding with Adaptive Fusion.
    - Uses GPE, Cosine, Sine, and Tanh embeddings.
    - Adapts fusion weights using local curvature or density.
    """
    def __init__(self, in_dim, out_dim, sigma, fusion_method="curvature"):
        super(EmbeddingMultiHybrid_3, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.fusion_method = fusion_method

        # Assuming you have existing implementations of these embedding functions:
        self.gpe = EmbeddingGPE(in_dim, out_dim, sigma)
        self.cosine = EmbeddingCosine(in_dim, out_dim)
        self.sine = EmbeddingSine(in_dim, out_dim)
        self.tanh = EmbeddingTanh(in_dim, out_dim)

    def forward(self, xyz, neighbor_xyz):
        emb_gpe = self.gpe(xyz)
        emb_cos = self.cosine(xyz)
        emb_sine = self.sine(xyz)
        emb_tanh = self.tanh(xyz)

        # Compute adaptive blend weights using the provided mode
        blend_weights = compute_blend_weights(neighbor_xyz, xyz, mode=self.fusion_method)

        # Apply weighted sum fusion (this example splits weights between GPE and the average of the other embeddings)
        fused_embedding = (blend_weights * emb_gpe +
                           (1 - blend_weights) * (emb_cos + emb_sine + emb_tanh) / 3)
        return fused_embedding



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


class AdaptiveFusionModule(nn.Module):
    def __init__(self, fusion_method="variance", eps=1e-6):
        super(AdaptiveFusionModule, self).__init__()
        self.fusion_method = fusion_method
        self.eps = eps

    def forward(self, embeddings, neighbor_xyz):
        # Compute statistics from neighbor_xyz:
        if neighbor_xyz.dim() == 4:
            stat_source = torch.mean(neighbor_xyz, dim=2)  # shape: [B, S, D']
        else:
            stat_source = neighbor_xyz  # shape: [B, N, D']

        if self.fusion_method == "variance":
            local_stat = torch.std(stat_source, dim=-1, keepdim=True)  # shape: [B, S, 1] or [B, N, 1]
        elif self.fusion_method == "curvature":
            local_stat = compute_local_curvature(neighbor_xyz)  # should return [B, S, 1] or [B, N, 1]
        else:
            raise ValueError("Unknown fusion method.")

        weights = 1.0 / (local_stat + self.eps)  # shape: [B, S, 1] or [B, N, 1]
        num_embeds = len(embeddings)
        weights = weights.expand(-1, -1, num_embeds)  # shape: [B, S, num_embeds] or [B, N, num_embeds]
        weights = torch.softmax(weights, dim=-1)

        fused_embedding = None
        for i, emb in enumerate(embeddings):
            if emb.dim() == 4:
                # For 4D embeddings, ensure weight_i has shape [B, S, 1, 1]
                weight_i = weights[..., i].unsqueeze(2).unsqueeze(-1)
            else:
                # For 3D embeddings, ensure weight_i has shape [B, N, 1]
                weight_i = weights[..., i].unsqueeze(-1)
            if fused_embedding is None:
                fused_embedding = emb * weight_i
            else:
                fused_embedding = fused_embedding + emb * weight_i
        return fused_embedding





class EmbeddingSine(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_features = math.ceil(out_dim / in_dim)
        self.out_idx = torch.linspace(0, self.num_features * in_dim - 1, out_dim).long()

    def forward(self, xyz):
        lin_vals = torch.linspace(-math.pi, math.pi, self.num_features + 2, device=xyz.device)[1:-1].reshape(1, -1)
        features = []
        for i in range(self.in_dim):
            tmp = xyz[..., i:i+1] - lin_vals
            features.append(torch.sin(tmp))
        feature = torch.cat(features, dim=-1)
        feature = torch.index_select(feature, -1, self.out_idx.to(xyz.device))
        return feature

class EmbeddingCosine(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_features = math.ceil(out_dim / in_dim)
        self.out_idx = torch.linspace(0, self.num_features * in_dim - 1, out_dim).long()

    def forward(self, xyz):
        lin_vals = torch.linspace(-math.pi, math.pi, self.num_features + 2, device=xyz.device)[1:-1].reshape(1, -1)
        features = []
        for i in range(self.in_dim):
            tmp = xyz[..., i:i+1] - lin_vals
            features.append(torch.cos(tmp))
        feature = torch.cat(features, dim=-1)
        feature = torch.index_select(feature, -1, self.out_idx.to(xyz.device))
        return feature

class EmbeddingTanh(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_features = math.ceil(out_dim / in_dim)
        self.out_idx = torch.linspace(0, self.num_features * in_dim - 1, out_dim).long()

    def forward(self, xyz):
        lin_vals = torch.linspace(-2.0, 2.0, self.num_features + 2, device=xyz.device)[1:-1].reshape(1, -1)
        features = []
        for i in range(self.in_dim):
            tmp = xyz[..., i:i+1] - lin_vals
            features.append(torch.tanh(tmp))
        feature = torch.cat(features, dim=-1)
        feature = torch.index_select(feature, -1, self.out_idx.to(xyz.device))
        return feature

class EmbeddingPolynomial(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = math.ceil(out_dim / in_dim)
        self.out_idx = torch.linspace(1, self.degree * in_dim, out_dim).long() - 1

    def forward(self, xyz):
        features = []
        for i in range(self.in_dim):
            for p in range(1, self.degree + 1):
                features.append(xyz[..., i:i+1] ** p)
        feature = torch.cat(features, dim=-1)
        feature = torch.index_select(feature, -1, self.out_idx.to(xyz.device))
        return feature

class EmbeddingLaplacian(nn.Module):
    def __init__(self, in_dim, out_dim, sigma):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.num_features = math.ceil(out_dim / in_dim)
        self.out_idx = torch.linspace(0, self.num_features * in_dim - 1, out_dim).long()
        self.feat_val = torch.linspace(-1.0, 1.0, self.num_features + 2)[1:-1].reshape(1, -1)

    def forward(self, xyz):
        features = []
        for i in range(self.in_dim):
            tmp = torch.abs(xyz[..., i:i+1] - self.feat_val.to(xyz.device))
            features.append(torch.exp(-tmp / self.sigma))
        feature = torch.cat(features, dim=-1)
        feature = torch.index_select(feature, -1, self.out_idx.to(xyz.device))
        return feature

class EmbeddingReciprocal(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_features = math.ceil(out_dim / in_dim)
        self.out_idx = torch.linspace(0, self.num_features * in_dim - 1, out_dim).long()
        self.offset = 1e-3

    def forward(self, xyz):
        features = []
        for i in range(self.in_dim):
            features.append(1.0 / (torch.abs(xyz[..., i:i+1]) + self.offset))
        feature = torch.cat(features, dim=-1)
        feature = torch.index_select(feature, -1, self.out_idx.to(xyz.device))
        return feature

class EmbeddingSinc(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_features = math.ceil(out_dim / in_dim)
        self.out_idx = torch.linspace(0, self.num_features * in_dim - 1, out_dim).long()
        self.pi = math.pi

    def forward(self, xyz):
        features = []
        for i in range(self.in_dim):
            tmp = xyz[..., i:i+1]
            numerator = torch.sin(self.pi * tmp)
            denominator = self.pi * tmp
            sinc = torch.where(torch.abs(tmp) < 1e-5, torch.ones_like(tmp), numerator / denominator)
            features.append(sinc)
        feature = torch.cat(features, dim=-1)
        feature = torch.index_select(feature, -1, self.out_idx.to(xyz.device))
        return feature



# New Multi-Hybrid Embedding Function: EmbeddingMultiHybrid
class EmbeddingMultiHybrid_1(nn.Module):
    """
    Multi-Hybrid Embedding Function

    This function computes several complementary embeddings (GPE, Cosine, Sine, Tanh)
    and fuses them using either concatenation + fixed projection or adaptive fusion.

    Args:
        in_dim (int): Input dimension (e.g., 3 for XYZ coordinates).
        out_dim (int): Desired output dimension.
        sigma (float): Kernel width for the GPE component.
        fusion_strategy (str): Fusion strategy: "concat" or "adaptive".
    """
    def __init__(self, in_dim, out_dim, sigma, fusion_strategy="concat"):
        super(EmbeddingMultiHybrid_1, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.fusion_strategy = fusion_strategy

        # We combine four embeddings: GPE, Cosine, Sine, and Tanh.
        # Divide the target dimension equally among them.
        self.n_embeds = 4
        import math
        self.embed_dim = math.ceil(out_dim / self.n_embeds)
        self.total_dim = self.n_embeds * self.embed_dim

        # Instantiate the individual embeddings.
        self.gpe = EmbeddingGPE(in_dim, self.embed_dim, sigma)
        self.cosine = EmbeddingCosine(in_dim, self.embed_dim)
        self.sine = EmbeddingSine(in_dim, self.embed_dim)
        self.tanh = EmbeddingTanh(in_dim, self.embed_dim)

        if fusion_strategy == "concat":
            # Create a fixed (non-learnable) projection: a linear transformation from total_dim to out_dim.
            projection_weight = torch.randn(self.total_dim, out_dim)
            projection_bias = torch.zeros(out_dim)
            self.register_buffer("projection_weight", projection_weight)
            self.register_buffer("projection_bias", projection_bias)
        elif fusion_strategy == "adaptive":
            # For adaptive fusion, we will compute per-embedding weights using local statistics.
            self.fusion_eps = 1e-6
        else:
            raise ValueError("fusion_strategy must be either 'concat' or 'adaptive'")

    def forward(self, xyz):
        # Compute each embedding. Each output is assumed to have shape [B, ..., embed_dim].
        emb_gpe = self.gpe(xyz)
        emb_cos = self.cosine(xyz)
        emb_sine = self.sine(xyz)
        emb_tanh = self.tanh(xyz)

        if self.fusion_strategy == "concat":
            # Concatenate the embeddings along the last dimension.
            cat_emb = torch.cat([emb_gpe, emb_cos, emb_sine, emb_tanh], dim=-1)
            # Apply the fixed projection.
            out = torch.matmul(cat_emb, self.projection_weight) + self.projection_bias
            return out

        elif self.fusion_strategy == "adaptive":
            eps = self.fusion_eps
            # Compute the standard deviation (as a proxy for local variation) along the feature dimension.
            std_gpe = torch.std(emb_gpe, dim=-1, keepdim=True)
            std_cos = torch.std(emb_cos, dim=-1, keepdim=True)
            std_sine = torch.std(emb_sine, dim=-1, keepdim=True)
            std_tanh = torch.std(emb_tanh, dim=-1, keepdim=True)
            # Invert these values to give higher weight to more stable embeddings.
            w_gpe = 1.0 / (std_gpe + eps)
            w_cos = 1.0 / (std_cos + eps)
            w_sine = 1.0 / (std_sine + eps)
            w_tanh = 1.0 / (std_tanh + eps)
            # Concatenate and normalize the weights via softmax.
            weights = torch.cat([w_gpe, w_cos, w_sine, w_tanh], dim=-1)  # shape: [B, ..., 4]
            weights = torch.softmax(weights, dim=-1)
            # Weight each embedding.
            emb_gpe_weighted = emb_gpe * weights[..., 0:1]
            emb_cos_weighted = emb_cos * weights[..., 1:2]
            emb_sine_weighted = emb_sine * weights[..., 2:3]
            emb_tanh_weighted = emb_tanh * weights[..., 3:4]
            # Fuse by summing the weighted embeddings.
            fusion_sum = emb_gpe_weighted + emb_cos_weighted + emb_sine_weighted + emb_tanh_weighted
            # If the fused embedding dimension differs from the desired output dimension, apply a fixed projection.
            if self.embed_dim != self.out_dim:
                if not hasattr(self, "adaptive_projection_weight"):
                    projection_weight = torch.randn(self.embed_dim, self.out_dim, device=xyz.device)
                    projection_bias = torch.zeros(self.out_dim, device=xyz.device)
                    self.register_buffer("adaptive_projection_weight", projection_weight)
                    self.register_buffer("adaptive_projection_bias", projection_bias)
                fusion_sum = torch.matmul(fusion_sum, self.adaptive_projection_weight) + self.adaptive_projection_bias
            return fusion_sum



class EmbeddingMultiHybrid_2(nn.Module):
    """
    Multi-Hybrid Embedding Function
    Computes complementary embeddings (GPE, Cosine, Sine, Tanh) and fuses them.
    Adjusts its behavior based on whether the input is 3D ([B, N, 3]) or 4D ([B, S, K, 3]).
    """
    def __init__(self, in_dim, out_dim, sigma, fusion_method="curvature"):
        super(EmbeddingMultiHybrid_2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.fusion_module = AdaptiveFusionModule(fusion_method)

        # Define sub-embeddings. They all use the same out_dim.
        self.gpe = EmbeddingGPE(in_dim, out_dim, sigma)
        self.cosine = EmbeddingCosine(in_dim, out_dim)
        self.sine = EmbeddingSine(in_dim, out_dim)
        self.tanh = EmbeddingTanh(in_dim, out_dim)

    def forward(self, xyz, neighbor_xyz):
        # Check input dimension:
        if xyz.dim() == 3:
            # Full point cloud case.
            emb_gpe = self.gpe(xyz)
            emb_cos = self.cosine(xyz)
            emb_sine = self.sine(xyz)
            emb_tanh = self.tanh(xyz)
        elif xyz.dim() == 4:
            # Local grouping case: Ensure each sub-embedding processes 4D input.
            emb_gpe = self.gpe(xyz)
            emb_cos = self.cosine(xyz)
            emb_sine = self.sine(xyz)
            emb_tanh = self.tanh(xyz)
        else:
            raise ValueError("Unsupported input dimension")
        
        # (Optionally) Check that all embeddings have the same shape.
        shape0 = emb_gpe.shape
        for emb in [emb_cos, emb_sine, emb_tanh]:
            if emb.shape != shape0:
                raise ValueError("All sub-embeddings must have the same output shape.")

        # Fuse the sub-embeddings adaptively.
        fused_embedding = self.fusion_module([emb_gpe, emb_cos, emb_sine, emb_tanh], neighbor_xyz)
        return fused_embedding




class EnrichedGeometricEmbedding(nn.Module):
    """
    Enriched Geometric Embedding

    This embedding function augments a standard Gaussian embedding with additional
    geometric cues (e.g., curvature and Laplacian features). It supports both 3D and 4D inputs.
    """
    def __init__(self, in_dim, out_dim, sigma, cues=["curvature", "laplacian"]):
        super(EnrichedGeometricEmbedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.cues = cues

        import math
        feat_dim = math.ceil(out_dim / (in_dim * (1 + len(cues))))
        self.base_feat_num = feat_dim * in_dim

        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)

        self.total_dim = self.base_feat_num
        if "curvature" in cues:
            self.total_dim += 1
        if "laplacian" in cues:
            self.total_dim += in_dim

        projection_weight = torch.randn(self.total_dim, out_dim)
        projection_bias = torch.zeros(out_dim)
        self.register_buffer("projection_weight", projection_weight)
        self.register_buffer("projection_bias", projection_bias)

    def forward(self, xyz, neighbor_xyz):
        """
        Args:
            xyz: Tensor of shape [B, N, in_dim] or [B, S, K, in_dim].
            neighbor_xyz: Tensor of the same shape as xyz.
        Returns:
            For 3D input: [B, N, out_dim]; for 4D input: [B, S, K, out_dim].
        """
        if xyz.dim() == 3:
            B, N, _ = xyz.shape
            embeds = []
            for i in range(self.in_dim):
                tmp = xyz[..., i:i+1] - self.feat_val.to(xyz.device)
                rbf = (-0.5 * tmp**2 / (self.sigma**2)).exp()
                embeds.append(rbf)
            gaussian_emb = torch.cat(embeds, dim=-1)
            all_features = [gaussian_emb]

            if "curvature" in self.cues:
                curvature = compute_local_curvature(neighbor_xyz)  # [B, N, 1]
                all_features.append(curvature)
            if "laplacian" in self.cues:
                neighbor_mean = torch.mean(neighbor_xyz, dim=2)  # [B, N, in_dim]
                laplacian = torch.abs(xyz - neighbor_mean)
                all_features.append(laplacian)

            fused_features = torch.cat(all_features, dim=-1)
            out = torch.matmul(fused_features, self.projection_weight) + self.projection_bias
            return out

        elif xyz.dim() == 4:
            B, S, K, _ = xyz.shape
            xyz_reshaped = xyz.view(B * S, K, self.in_dim)
            neighbor_xyz_reshaped = neighbor_xyz.view(B * S, K, self.in_dim)

            embeds = []
            for i in range(self.in_dim):
                tmp = xyz_reshaped[..., i:i+1] - self.feat_val.to(xyz.device)
                rbf = (-0.5 * tmp**2 / (self.sigma**2)).exp()
                embeds.append(rbf)
            gaussian_emb = torch.cat(embeds, dim=-1)
            all_features = [gaussian_emb]

            if "curvature" in self.cues:
                curvature = compute_local_curvature(neighbor_xyz_reshaped)  # shape: [B*S, 1]
                curvature = curvature.unsqueeze(1)  # Now shape becomes [B*S, 1, 1]
                curvature = curvature.expand(-1, K, -1)  # Now shape becomes [B*S, K, 1]

                # Expand curvature to have the same K dimension:
                curvature = curvature.expand(-1, K, -1)  # shape: [B*S, K, 1]
                all_features.append(curvature)
            if "laplacian" in self.cues:
                neighbor_mean = torch.mean(neighbor_xyz_reshaped, dim=1, keepdim=True)  # [B*S, 1, in_dim]
                laplacian = torch.abs(xyz_reshaped - neighbor_mean)  # [B*S, K, in_dim]
                all_features.append(laplacian)

            fused_features = torch.cat(all_features, dim=-1)  # [B*S, K, total_dim]
            out = torch.matmul(fused_features, self.projection_weight) + self.projection_bias  # [B*S, K, out_dim]
            out = out.view(B, S, K, self.out_dim)
            return out

        else:
            raise ValueError("Unsupported input dimension for EnrichedGeometricEmbedding")




# Helper function for computing Laplacian features (if needed separately)
def compute_laplacian_features(xyz, neighbor_xyz):
    """
    Computes the Laplacian feature, defined as the absolute difference between
    each point and the mean of its neighbors.
    
    Args:
        xyz: Tensor of shape [B, N, in_dim] (points).
        neighbor_xyz: Tensor of shape [B, N, K, in_dim] (neighbors).
    Returns:
        Tensor of shape [B, N, in_dim] representing the Laplacian features.
    """
    neighbor_mean = torch.mean(neighbor_xyz, dim=2)  # [B, N, in_dim]
    laplacian = torch.abs(xyz - neighbor_mean)
    return laplacian


# New Fusion Module for Embeddings: EmbeddingFusionModule
class EmbeddingFusionModule(nn.Module):
    """
    Fusion Module for Embeddings

    This module fuses multiple embedding outputs using a specified element-wise operation.
    Supported fusion methods are "sum", "product", or "max".
    """
    def __init__(self, fusion_method="sum"):
        super(EmbeddingFusionModule, self).__init__()
        self.fusion_method = fusion_method

    def forward(self, embeddings):
        # 'embeddings' should be a list of tensors of identical shape.
        if not embeddings:
            raise ValueError("No embeddings provided for fusion.")
        if self.fusion_method == "sum":
            fused = embeddings[0]
            for emb in embeddings[1:]:
                fused = fused + emb
        elif self.fusion_method == "product":
            fused = embeddings[0]
            for emb in embeddings[1:]:
                fused = fused * emb
        elif self.fusion_method == "max":
            # Stack along a new dimension and take element-wise maximum.
            stacked = torch.stack(embeddings, dim=-1)
            fused, _ = torch.max(stacked, dim=-1)
        else:
            raise ValueError("Unknown fusion method. Choose 'sum', 'product', or 'max'.")
        return fused





class NeighborPooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(nn.GELU())

    def forward(self, feat_knn_w):
        feat_agg = feat_knn_w.mean(-2) + feat_knn_w.max(-2)[0]
        feat_agg = self.out_transform(feat_agg.transpose(-2, -1))
        return feat_agg.transpose(-2, -1)

class EncoderGPECls(nn.Module):
    def __init__(self, num_points, init_dim, stages, stage_dim, k, sigma, feat_normalize, embedding_fn="gpe", **kwargs):
        super().__init__()
        self.num_points = num_points
        self.init_dim = init_dim
        self.stages = stages
        self.stage_dim = stage_dim
        self.sigma = sigma
        self.k = k

        embedding_map = {
            "gpe": EmbeddingGPE,
            "sine": EmbeddingSine,
            "cosine": EmbeddingCosine,
            "tanh": EmbeddingTanh,
            "poly": EmbeddingPolynomial,
            "lap": EmbeddingLaplacian,
            "recip": EmbeddingReciprocal,
            "sinc": EmbeddingSinc,
            "hybrid": EmbeddingHybridAugmented,  # New hybrid function
            "gausslap": EmbeddingGaussianLaplacianFusion, #
            "adaptive_1": AdaptiveEmbeddingGPE_1, 
            "adaptive_2": AdaptiveEmbeddingGPE_2,
            "multihybrid1": EmbeddingMultiHybrid_1, #
            "multihybrid2": EmbeddingMultiHybrid_2, #
            "multihybrid3": EmbeddingMultiHybrid_3,
            "geo": EnrichedGeometricEmbedding,      #
            # If you have an entry for adaptive fusion, add it here.
        }
        self.embed_cls = embedding_map.get(embedding_fn) #, EmbeddingGPE)
        # Pass extra keyword arguments (e.g., blend, fusion_method, eps, etc.) to the embedding constructor.
        #self.init_embed = self.embed_cls(3, self.init_dim, sigma, **kwargs)
        # Define which arguments should be passed for each embedding type
        valid_args = {
            "hybrid": ["sigma", "blend"],
            "gausslap": ["sigma", "blend"],
            "adaptive_1": ["sigma"],
            "adaptive_2": ["base_sigma", "blend_strategy", "eps"],
            "multihybrid1": ["sigma"],
            "multihybrid2": ["sigma"],  # ✅ FIX: Removed "blend_strategy"
            "multihybrid3": ["sigma", "fusion_method"],
            "geo": ["sigma", "cues"],
        }



        # Get the correct key for the embedding function
        embedding_fn_key = embedding_fn.lower()
        allowed_keys = valid_args.get(embedding_fn_key, ["sigma"])  # Default to only sigma

        # Remove unrecognized arguments
        embedding_args = {k: v for k, v in kwargs.items() if k in allowed_keys and v is not None}
        
        # For adaptive_2, use "base_sigma" instead of "sigma"
        if embedding_fn_key == "adaptive_2":
            embedding_args["base_sigma"] = sigma
        else:
            embedding_args["sigma"] = sigma

        # Initialize embedding function with the correct arguments
        self.init_embed = self.embed_cls(3, self.init_dim, **embedding_args)




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
        # feat = self.init_embed(xyz)
        if isinstance(self.init_embed, (EmbeddingMultiHybrid_2, EnrichedGeometricEmbedding, AdaptiveEmbeddingGPE_2, EmbeddingMultiHybrid_3)):
            neighbor_xyz = compute_neighbors(xyz, k=self.k)  # Ensure k-NN computation
            feat = self.init_embed(xyz, neighbor_xyz)  # Pass neighbor_xyz
        else:
            feat = self.init_embed(xyz)  # Standard call




        stage_results = []
        for i in range(self.stages):
            xyz, feat, xyz_knn, feat_knn = self.lg_list[i](xyz, feat)
            feat_knn_w = self.agpe_list[i](xyz_knn, feat_knn)
            feat = self.pool_list[i](feat_knn_w)
            stage_pooling = torch.cat((feat.max(-2)[0], feat.mean(-2)), dim=1)
            stage_results.append(stage_pooling)
        encoded_out = torch.cat(stage_results, dim=1)
        return encoded_out




'''
class PointGNCls(nn.Module):
    def __init__(self, num_points=1024, init_dim=6, stages=4, stage_dim=72, k=90, sigma=0.3, feat_normalize=True, embedding_fn="gpe"):
        super().__init__()
        self.EncNP = EncoderGPECls(num_points, init_dim, stages, stage_dim, k, sigma, feat_normalize, embedding_fn)

    def forward(self, xyz):
        x = self.EncNP(xyz)
        return x
'''

class PointGNCls(nn.Module):
    def __init__(self, num_points=1024, init_dim=6, stages=4, stage_dim=72, k=90, sigma=0.3, feat_normalize=True, embedding_fn="gpe", **kwargs):
        super().__init__()
        self.EncNP = EncoderGPECls(num_points, init_dim, stages, stage_dim, k, sigma, feat_normalize, embedding_fn, **kwargs)

    def forward(self, xyz):
        x = self.EncNP(xyz)
        return x



if __name__ == "__main__":
    import time
    batch_size = 128
    num_points = 1024
    in_ch = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    random_xyz = torch.randn(batch_size, num_points, in_ch).contiguous().to(device)
    model = PointGNCls(num_points=num_points).to(device)
    start_time = time.time()
    output = model(random_xyz)
    end_time = time.time()
    print(f"Output shape: {output.shape}")
    print(f"Time taken for forward pass: {end_time - start_time:.6f} seconds")
