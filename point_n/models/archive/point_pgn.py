# Parametric Networks for 3D Point Cloud Classification
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
    def __init__(self, group_num, k_neighbors,type):
        super().__init__()
        self.type = type
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
        
        '''
        # Normalize x (features) and xyz (coordinates)
        center_xyz = lc_xyz.unsqueeze(dim=-2)  # [B, G, 1, 3]
        std_xyz = torch.std(knn_xyz - center_xyz)  # [1]
        knn_xyz = (knn_xyz - center_xyz) / (std_xyz + 1e-5)  # [B, G, K, 3]

        center_x = lc_x.unsqueeze(dim=-2)  # [B, G, 1, D]
        std_x = torch.std(knn_x - center_x)  # [1]
        knn_x = (knn_x - center_x) / (std_x + 1e-5)  # [B, G, K, D]
        '''
        
        # Normalization
        if self.type == 'mn40':
            mean_xyz = lc_xyz.unsqueeze(dim=-2)
            std_xyz = torch.std(knn_xyz - mean_xyz)
            knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        elif self.type == 'scan':
            knn_xyz = knn_xyz.permute(0, 3, 1, 2)
            knn_xyz -= lc_xyz.permute(0, 2, 1).unsqueeze(-1)
            knn_xyz /= torch.abs(knn_xyz).max(dim=-1, keepdim=True)[0]
            knn_xyz = knn_xyz.permute(0, 2, 3, 1)

        # Feature Expansion
        b, g, k, _ = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(b, g, 1, -1).repeat(1, 1, k, 1)], dim=-1)

        return lc_xyz, lc_x, knn_xyz, knn_x


# GPE Aggregation
class AggregationGPE(nn.Module):
    def __init__(self, out_dim, sigma, block_num, dim_expansion):
        super().__init__()
        self.geo_extract = EmbeddingGPE(3, out_dim, sigma)
        
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1
        self.linear1 = Linear1Layer(out_dim * expand, out_dim, bias=False)
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(Linear2Layer(out_dim, bias=True))
        self.linear2 = nn.Sequential(*self.linear2)


    def forward(self, knn_xyz, knn_x):

        B, G, K, C = knn_x.shape
        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x = self.linear1(knn_x.reshape(B, -1, G*K)).reshape(B, -1, G, K)

        # Geometry Extraction
        position_embed=self.geo_extract(knn_xyz)
        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        return lc_x
    

# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels/2),
                    kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(in_channels/2)),
            self.act
        )
        self.net2 = nn.Sequential(
                nn.Conv2d(in_channels=int(in_channels/2), out_channels=in_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm2d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)
    

# PosE for Local Geometry Extraction

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

   
# Parametric Encoder
class EncP(nn.Module):  
    def __init__(self, in_channels, input_points, num_stages, embed_dim, k_neighbors, sigma, AggregationGPE_block, dim_expansion, type):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.sigma=sigma

        # Raw-point Embedding
        self.raw_point_embed = Linear1Layer(in_channels, self.embed_dim, bias=False)

        self.Local_Grouper_list = nn.ModuleList() # FPS, kNN
        self.AggregationGPE_list = nn.ModuleList() # GPE Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]
            group_num = group_num // 2
            self.Local_Grouper_list.append(Local_Grouper(group_num, k_neighbors,type))
            self.AggregationGPE_list.append(AggregationGPE(out_dim, self.sigma, AggregationGPE_block[i], dim_expansion[i]))
            self.Pooling_list.append(Pooling(out_dim))


    def forward(self, xyz, x):

        # Raw-point Embedding
        # pdb.set_trace()
        x = self.raw_point_embed(x)

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.Local_Grouper_list[i](xyz, x.permute(0, 2, 1))
            # GPE Aggregation
            knn_x_w = self.AggregationGPE_list[i]( knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)
        return x


# Parametric Network for ModelNet40
class Point_PGN_mn40(nn.Module):
    def __init__(self, in_channels=3, class_num=40, input_points=1024, num_stages=4, embed_dim=36, k_neighbors=40, sigma=0.3, AggregationGPE_block=[2,1,1,1], dim_expansion=[2,2,2,1], type='mn40'):
        super().__init__()
        # Parametric Encoder
        self.EncP = EncP(in_channels, input_points, num_stages, embed_dim, k_neighbors,sigma, AggregationGPE_block, dim_expansion, type)
        self.out_channel = embed_dim
        for i in dim_expansion:
            self.out_channel *= i
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, class_num)
        )


    def forward(self, x):
        # xyz: point coordinates
        # x: point features
        xyz = x.permute(0, 2, 1)

        # Parametric Encoder
        x = self.EncP(xyz, x)

        # Classifier
        x = self.classifier(x)
        return x
    

# Parametric Network for ScanObjectNN
class Point_PGN_scan(nn.Module):
    def __init__(self, in_channels=4, class_num=15, input_points=1024, num_stages=4, embed_dim=36, k_neighbors=40, sigma=0.3, AggregationGPE_block=[2,1,1,1], dim_expansion=[2,2,2,1], type='scan'):
        super().__init__()
        # Parametric Encoder
        self.EncP = EncP(in_channels, input_points, num_stages, embed_dim, k_neighbors,sigma, AggregationGPE_block, dim_expansion, type)
        self.out_channel = embed_dim
        for i in dim_expansion:
            self.out_channel *= i
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, class_num)
        )


    def forward(self, x, xyz):
        # xyz: point coordinates
        # x: point features

        # Parametric Encoder
        x = self.EncP(xyz, x)

        # Classifier
        x = self.classifier(x)
        return x
