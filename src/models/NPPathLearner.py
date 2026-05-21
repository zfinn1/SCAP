
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils

def knn_point(k, xyz, new_xyz):
    # xyz: [B, N, 3], new_xyz: [B, M, 3]
    dist = torch.cdist(new_xyz, xyz)  # [B, M, N]
    idx = dist.topk(k, largest=False)[1]  # [B, M, k]
    return idx

def index_points(points, idx):
    # points: [B, N, C], idx: [B, M, k]
    B = points.shape[0]
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1)
    return points[batch_indices, idx, :]



# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim=3, out_dim=256, alpha=100, beta=1000):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz):
        B, N, K, _ = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim, device=knn_xyz.device).float()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)

        div_embed = (self.beta * knn_xyz.unsqueeze(-1)) / dim_embed
        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)

        position_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [B,N,K,3,2*feat_dim]
        position_embed = position_embed.flatten(3)  # [B,N,K, 3*2*feat_dim] = [B,N,K,252]

        # --- padding 到 256 ---
        if position_embed.shape[-1] < self.out_dim:
            pad_dim = self.out_dim - position_embed.shape[-1]
            pad = torch.zeros(B, N, K, pad_dim, device=knn_xyz.device)
            position_embed = torch.cat([position_embed, pad], dim=-1)  # [B,N,K,256]

        return position_embed


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def forward(self, knn_x_w):
        # knn_x_w: [B,N,k,C]
        lc_x = knn_x_w.max(2)[0] + knn_x_w.mean(2)  # [B,N,C]
        lc_x = self.out_transform(lc_x.transpose(1,2)).transpose(1,2)
        return lc_x

class NPPathLearner(nn.Module):
    def __init__(self, out_dim=256, k_neighbors=16, alpha=100, beta=1000):
        super().__init__()
        self.k = k_neighbors
        self.geo_extract = PosE_Geo(in_dim=3, out_dim=out_dim, alpha=alpha, beta=beta)
        self.pooling = Pooling(out_dim)

        self.proto_attn = nn.MultiheadAttention(
        embed_dim=out_dim,
        num_heads=4,
        batch_first=True
    )
    def forward(self, src_proto, tgt_proto, src_xyz, tgt_xyz):
        """
        src_proto: [B, Ns, C]
        tgt_proto: [B, Nt, C]
        src_xyz:   [B, Ns, 3]
        tgt_xyz:   [B, Nt, 3]
        return: src_aligned: [B, Ns, C]
        """
        B, Ns, C = src_proto.shape
        _, Nt, _ = tgt_proto.shape

        # kNN 在 target xyz 上找 src 的邻居
        knn_idx = knn_point(self.k, tgt_xyz, src_xyz)   # [B, Ns, k]
        knn_xyz = index_points(tgt_xyz, knn_idx)        # [B, Ns, k, 3]

        # 非参数化位置编码
        position_embed = self.geo_extract(knn_xyz)      # [B, Ns, k, C]

        # 聚合 (pooling)
        lc_x = self.pooling(position_embed)             # [B, Ns, C]

        # proto 交互（注意力）
        weights = torch.softmax(torch.einsum("bnc,bmc->bnm", src_proto, tgt_proto), dim=-1)  # [B,Ns,Nt]
        src_aligned = torch.einsum("bnm,bmc->bnc", weights, tgt_proto) + lc_x                # [B,Ns,C]

        return src_aligned



# def knn_point(k, xyz, new_xyz):
#     # xyz: [B, N, C], new_xyz: [B, M, C]
#     dist = torch.cdist(new_xyz, xyz)  # [B, M, N]
#     idx = dist.topk(k, largest=False)[1]  # [B, M, k]
#     return idx
#
#
# # PosE for Local Geometry Extraction
# class PosE_Geo(nn.Module):
#     def __init__(self, in_dim, out_dim, alpha, beta):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.alpha, self.beta = alpha, beta
#
#     def forward(self, knn_xyz, knn_x):
#         B, _, G, K = knn_xyz.shape# 4 16 16 256
#         feat_dim = self.out_dim // (self.in_dim * 2)
#
#         feat_range = torch.arange(feat_dim).float().cuda()
#         dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
#         div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)
#
#         sin_embed = torch.sin(div_embed)
#         cos_embed = torch.cos(div_embed)
#         position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
#         position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)
#
#         # Weigh
#         knn_x_w = knn_x + position_embed
#         knn_x_w *= position_embed
#
#         return knn_x_w
#
# # Pooling
# class Pooling(nn.Module):
#     def __init__(self, out_dim):
#         super().__init__()
#         self.out_transform = nn.Sequential(
#                 nn.BatchNorm1d(out_dim),
#                 nn.GELU())
#
#     def forward(self, knn_x_w):
#         # Feature Aggregation (Pooling)
#         lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
#         lc_x = self.out_transform(lc_x)
#         return lc_x
#
# def index_points(points, idx):
#     # points: [B, N, C], idx: [B, M, k]
#     B = points.shape[0]
#     batch_indices = torch.arange(B, device=points.device).view(B, 1, 1)
#     return points[batch_indices, idx, :]
#
# class NPPathLearner(nn.Module):
#     def __init__(self, out_dim, k_neighbors=16, alpha=100, beta=1000):
#         super().__init__()
#         self.k = k_neighbors
#         self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)
#         self.pooling = Pooling(out_dim)
#
#     def forward(self, src_proto, tgt_proto, src_xyz, tgt_xyz):
#         """
#         src_proto: [B, N, C] 源点云原型
#         tgt_proto: [B, M, C] 目标点云原型
#         return: src_aligned: [B, N, C]
#         """
#         B, Np, C = src_proto.shape
#         _, Mp, _ = tgt_proto.shape
#
#         # # kNN: 每个源原型在目标原型中找邻居
#         # knn_idx = knn_point(self.k, tgt_proto, src_proto)  # [B, N, k]
#         # knn_xyz = index_points(tgt_proto, knn_idx)  # [B, N, k, C]
#         # knn_x = knn_xyz.clone()  # 简化：直接用 tgt_proto 特征
#         #
#         # # Local Geometry Aggregation (Point-NN思想)
#         #
#         # #16  607  16  256
#         # knn_x_w = self.geo_extract(knn_xyz, knn_x)
#         #
#         # # knn_x_w = self.geo_extract(lc_xyz.unsqueeze(1), lc_x.unsqueeze(1), knn_xyz, knn_x)
#         #
#         # # Pooling 聚合 → 得到对齐后的特征
#         # src_aligned = self.pooling(knn_x_w).transpose(1, 2)  # [B, N, C]
#
#
#         # --- 关键修改：基于坐标做 knn ---
#
#         knn_idx = knn_point(self.k, tgt_xyz, src_xyz)  # [B, Ns, k]
#         knn_xyz = index_points(tgt_xyz, knn_idx)  # [B, Ns, k, 3]
#         knn_x = index_points(tgt_xyz, knn_idx)  # 用坐标特征 (或者额外的点云特征)，而不是 proto
#
#         # Step2: 聚合 (Point-NN)
#         knn_xyz = knn_xyz.permute(0, 3, 1, 2)  # [B, 3, Ns, k]
#         knn_x = knn_x.permute(0, 3, 1, 2)  # [B, 3, Ns, k] 这里我们简单用坐标作为特征
#         knn_x_w = self.geo_extract(knn_xyz, knn_x)  # [B, C, Ns, k]
#
#         # Pooling 得到增强点特征
#         src_points_enhanced = self.pooling(knn_x_w).transpose(1, 2)  # [B, Ns, C]
#
#         # Step4: 把增强的点特征映射到 proto (用注意力 / pooling)
#         # 这里选择点到 proto 的加权平均
#         weights = torch.softmax(torch.einsum("bnc,bmc->bnm", src_proto, src_points_enhanced), dim=-1)  # [B, Np, Ns]
#         src_aligned = torch.einsum("bnm,bmc->bnc", weights, src_points_enhanced)  # [B, N
#
#         return src_aligned
