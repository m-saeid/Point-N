import torch
import math
0,2,3
0,3
0

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FourierFeatureEncoding, self).__init__()
        self.B = torch.randn(in_dim, out_dim // 2) * 2 * math.pi

    def forward(self, x):
        x_proj = x @ self.B.to(x.device)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    
    class GaussianRandomFeatures(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=1.0):
        super(GaussianRandomFeatures, self).__init__()
        self.W = torch.randn(in_dim, out_dim) / sigma

    def forward(self, x):
        projection = x @ self.W.to(x.device)
        return torch.cos(projection)
    
    
class PolynomialEncoding(nn.Module):
    def __init__(self, degree):
        super(PolynomialEncoding, self).__init__()
        self.degree = degree

    def forward(self, x):
        out = [x ** i for i in range(1, self.degree + 1)]
        return torch.cat(out, dim=-1)


import torch
import torch.nn as nn
import numpy as np
from scipy.special import sph_harm

class SphericalHarmonicsEncoding(nn.Module):
    def __init__(self, max_degree):
        super(SphericalHarmonicsEncoding, self).__init__()
        self.max_degree = max_degree

    def forward(self, coords):
        """
        Args:
            coords: Tensor of shape (..., 3), representing (x, y, z) coordinates.
        Returns:
            Tensor of shape (..., num_harmonics), where num_harmonics depends on max_degree.
        """
        # Normalize coordinates to lie on the unit sphere
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        norm = torch.sqrt(x**2 + y**2 + z**2) + 1e-8  # Avoid division by zero
        x, y, z = x / norm, y / norm, z / norm

        # Convert Cartesian coordinates to spherical coordinates
        theta = torch.acos(z)        # Inclination angle [0, pi]
        phi = torch.atan2(y, x)      # Azimuthal angle [-pi, pi]

        # Prepare lists to collect harmonics
        harmonics = []

        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                # Compute the spherical harmonic Y_l^m(theta, phi)
                Y_lm = self.compute_spherical_harmonic(l, m, theta, phi)
                harmonics.append(Y_lm)

        # Stack all harmonics along the last dimension
        harmonics = torch.stack(harmonics, dim=-1)
        return harmonics

    def compute_spherical_harmonic(self, l, m, theta, phi):
        """
        Computes the real part of the spherical harmonic function Y_l^m.
        """
        # sph_harm returns complex numbers; we take the real part
        sph_val = sph_harm(m, l, phi.cpu().numpy(), theta.cpu().numpy())
        sph_val = np.real(sph_val)
        return torch.from_numpy(sph_val).to(theta.device).float()
    
    
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

class LaplacePositionalEncoding(nn.Module):
    def __init__(self, num_eigenvectors):
        super(LaplacePositionalEncoding, self).__init__()
        self.num_eigenvectors = num_eigenvectors

    def forward(self, adjacency_matrix):
        """
        Args:
            adjacency_matrix: Sparse or dense adjacency matrix of shape (N, N).
        Returns:
            Tensor of shape (N, num_eigenvectors) containing the Laplacian eigenvectors.
        """
        # Ensure the adjacency matrix is in CSR format
        if not sp.issparse(adjacency_matrix):
            adjacency_matrix = sp.csr_matrix(adjacency_matrix)

        # Compute the degree matrix
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
        D = sp.diags(degrees)

        # Compute the unnormalized Laplacian matrix L = D - A
        L = D - adjacency_matrix

        # Compute the k smallest eigenvalues and corresponding eigenvectors
        # Since L is symmetric positive semi-definite, we use 'SM' (smallest magnitude)
        eigvals, eigvecs = splinalg.eigsh(L, k=self.num_eigenvectors, which='SM')

        # Convert eigenvectors to torch tensor
        eigvecs = torch.from_numpy(eigvecs).float()

        return eigvecs



class NonParametricAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_knn_w):
        # feat_knn_w: [B, S, K, D]
        # Compute attention scores
        attention_scores = feat_knn_w.mean(dim=-1, keepdim=True)  # [B, S, K, 1]
        attention_scores = torch.softmax(attention_scores, dim=2)  # [B, S, K, 1]
        # Apply attention
        feat_knn_w = feat_knn_w * attention_scores  # [B, S, K, D]
        return feat_knn_w



class AggregationGPE(nn.Module):
    def __init__(self, out_dim, sigma):
        super().__init__()
        self.gpe_embed = EmbeddingGPE(3, out_dim, sigma)
        self.attention = NonParametricAttention()

    def forward(self, xyz_knn, feat_knn):
        # Geometry Extraction
        position_embed = self.gpe_embed(xyz_knn)  # [B, S, K, out_dim]

        # Weigh
        feat_knn_w = feat_knn + position_embed  # [B, S, K, out_dim]
        feat_knn_w *= position_embed  # [B, S, K, out_dim]

        # Apply Non-Parametric Attention
        feat_knn_w = self.attention(feat_knn_w)  # [B, S, K, out_dim]

        return feat_knn_w  # [B, S, K, out_dim]


class PointGNCls(nn.Module):
    def __init__(
        self,
        num_points=1024,
        init_dim=6,
        stages=4,
        stage_dim=72,
        k=90,
        sigma=0.3,
        num_classes=40,  # Assuming 40 classes for classification
    ):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncoderGPECls(
            num_points,
            init_dim,
            stages,
            stage_dim,
            k,
            sigma,
        )
        # Calculate the total dimension after concatenation
        self.total_dim = 0
        out_dim = init_dim if init_dim != 0 else 3
        for i in range(stages):
            out_dim *= 2
            self.total_dim += out_dim * 2  # Since we concatenate max and mean pooling

        # Classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.total_dim),
            nn.Dropout(0.5),
            nn.Linear(self.total_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, xyz):
        # xyz: point coordinates # [B, N, 3]

        # Non-Parametric Encoder
        x = self.EncNP(xyz)  # [B, total_dim]

        # Classification
        x = self.classifier(x)  # [B, num_classes]
        return x


# Non-Parametric Feature Encoder with Skip Connections
class EncoderGPECls(nn.Module):
    def __init__(
        self,
        num_points,
        init_dim,
        stages,
        stage_dim,
        k,
        sigma,
    ):
        super().__init__()
        self.num_points = num_points
        self.init_dim = init_dim
        self.stages = stages
        self.stage_dim = stage_dim
        self.sigma = sigma

        # Initial Embedding
        self.init_embed = EmbeddingGPE(3, self.init_dim, sigma)

        self.lg_list = nn.ModuleList()  # FPS, kNN
        self.agpe_list = nn.ModuleList()  # GPE Aggregation
        self.pool_list = nn.ModuleList()  # Pooling

        out_dim = self.init_dim if self.init_dim != 0 else 3
        stage_points = self.num_points

        # Multi-stage Hierarchy
        for i in range(self.stages):
            out_dim = out_dim * 2
            stage_points = stage_points // 2
            self.lg_list.append(LocalGrouper(stage_points, k))
            self.agpe_list.append(AggregationGPE(out_dim, self.sigma))
            self.pool_list.append(NeighborPooling(out_dim))

    def forward(self, xyz):
        # xyz: point coordinates    # [B, N, 3]

        # Initial Embedding
        feat = self.init_embed(xyz)  # [B, N, init_dim]

        stage_results = []
        skip_feat = feat  # For skip connections

        # Multi-stage Hierarchy
        for i in range(self.stages):
            # FPS, kNN
            xyz, feat, xyz_knn, feat_knn = self.lg_list[i](xyz, feat)

            # GPE Aggregation
            feat_knn_w = self.agpe_list[i](xyz_knn, feat_knn)

            # Neighbor Pooling
            feat = self.pool_list[i](feat_knn_w)  # [B, N/2^i, D_i * 2]

            # Skip Connection: concatenate with features from previous stage
            if i > 0:
                # Ensure dimensions match for concatenation
                if skip_feat.shape[1] != feat.shape[1]:
                    # If number of points is different, we can interpolate or sample
                    # Here, we use interpolation for simplicity
                    skip_feat = nn.functional.interpolate(skip_feat.transpose(1, 2), size=feat.shape[1]).transpose(1, 2)
                feat = feat + skip_feat  # Skip connection (element-wise addition)
            skip_feat = feat  # Update skip_feat for the next stage

            # Stage Pooling
            stage_pooling = torch.cat((feat.max(-2)[0], feat.mean(-2)), dim=1)
            stage_results.append(stage_pooling)

        # Concatenate features from all stages
        encoded_out = torch.cat(stage_results, dim=1)  # [B, total_dim]

        return encoded_out
