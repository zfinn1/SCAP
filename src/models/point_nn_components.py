# models/nonparametric.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils


def knn_point(k, xyz, new_xyz):
    dist = torch.cdist(new_xyz, xyz)
    idx = dist.topk(k, dim=-1, largest=False)[1]
    return idx


def index_points(points, idx):
    B, N, C = points.shape
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


class PosE_Geo(nn.Module):
    """Point-NN的位置编码模块"""

    def __init__(self, in_dim, out_dim, alpha=100, beta=1000):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.beta = beta

    def forward(self, knn_xyz):
        B, G, K, _ = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim, dtype=torch.float32, device=knn_xyz.device)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.cat([sin_embed, cos_embed], dim=-1)
        position_embed = position_embed.view(B, G, K, -1).permute(0, 3, 1, 2)

        return position_embed


class NonParametricAlignment(nn.Module):
    """Point-NN核心对齐模块"""

    def __init__(self, out_dim, k_neighbors=16, alpha=100, beta=1000):
        super().__init__()
        self.k = k_neighbors
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)

    def forward(self, src_points, tgt_points):
        """
        src_points: [B, N, C] 源点云原型
        tgt_points: [B, M, C] 目标点云原型
        return: [B, N, C] 对齐后的特征
        """
        # 1. 寻找k最近邻
        knn_idx = knn_point(self.k, tgt_points, src_points)  # [B, N, k]
        knn_xyz = index_points(tgt_points, knn_idx)  # [B, N, k, C]

        # 2. 计算局部参考点 (质心)
        centroid = knn_xyz.mean(dim=2, keepdim=True)  # [B, N, 1, C]

        # 3. 位置归一化
        knn_xyz_norm = knn_xyz - centroid

        # 4. 位置编码 (Point-NN核心)
        position_embed = self.geo_extract(knn_xyz_norm)  # [B, C, N, k]

        # 5. 聚合特征 (最大池化+平均池化)
        max_feat = position_embed.max(dim=-1)[0]  # [B, C, N]
        mean_feat = position_embed.mean(dim=-1)  # [B, C, N]
        aggregated = max_feat + mean_feat  # [B, C, N]

        return aggregated.permute(0, 2, 1)  # [B, N, C]