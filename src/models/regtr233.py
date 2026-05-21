"""REGTR network architecture
"""
import math

import torch
import torch.nn as nn

from models.backbone_kpconv.kpconv import KPFEncoder, PreprocessorGPU, compute_overlaps
from models.generic_reg_model2 import GenericRegModel
from models.losses.corr_loss import CorrCriterion
from models.losses.feature_loss import InfoNCELossFull, CircleLossFull
from models.transformer.position_embedding import PositionEmbeddingCoordsSine, \
    PositionEmbeddingLearned
from models.transformer.transformers import \
    TransformerCrossEncoderLayer, TransformerCrossEncoder
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_inv
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences
from utils.viz2 import save_registration_mat
from utils.viz3 import visualize_registration_offline

_TIMEIT = False

from models.NPPathLearner2 import NPPathLearner


import torch.nn.functional as F
class AlignmentDrivenTransformerLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src_feats, tgt_feats, aligned_src_proto):
        """
        src_feats: [B, N, C] 4  607 256
        tgt_feats: [B, M, C]    4  592 256
        aligned_src_proto: [B, P, C] 4 128  256
        """
        # self-attn
        src2, _ = self.self_attn(src_feats, src_feats, src_feats)
        src_feats = self.norm1(src_feats + self.dropout(src2))

        # cross-attn
        src2, _ = self.cross_attn(src_feats, tgt_feats, tgt_feats)

        # prototype gating
        gate = self.sigmoid(self.gate_proj(aligned_src_proto.mean(dim=1)))  # [B, C]
        gate = gate.unsqueeze(1).expand_as(src2)
        src2 = gate * src2

        src_feats = self.norm2(src_feats + self.dropout(src2))

        # FFN
        ff = self.ffn(src_feats)
        src_feats = self.norm3(src_feats + self.dropout(ff))

        return src_feats



class SemanticPrototypeExtractor(nn.Module):
    def __init__(self, in_dim=256, num_prototypes=8, num_heads=8):
        super().__init__()
        # 学习型 prototype token
        self.prototype_tokens = nn.Parameter(torch.randn(num_prototypes, in_dim))
        # 用标准多头注意力，让 prototypes 从点特征中吸收信息
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=False)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, feats):
        """
        Args:
            feats: [B, N, C]  点特征 (来自 transformer encoder 输出)
        Returns:
            proto_out: [B, P, C]  结构语义原型
语义原型提取器
作用
压缩点级特征到“原型”：
原始点特征太多（每帧几百/几千个点），噪声大，不便直接做对齐。
num_prototypes（比如 64）就是从点云中抽象出来的“结构语义原型”，它们可以看作是点云的语义中心。
多头注意力聚合：
用 prototype_tokens 作为 Query，点特征作为 Key/Value。
每个 prototype 会从所有点中“吸收”信息 → 得到一组稳定的高层表达。
作用效果
降维：把 [N, B, C] 的点集，压缩成 [B, P, C]（P ≪ N）。
去噪：原型能滤掉孤立点的干扰。
结构建模：原型学到的是局部结构或语义中心，有利于后续对齐。
总结：
结构语义原型提取器的主要作用是将稠密点特征压缩为少量结构语义原型，以增强对齐的稳定性和泛化能力。
        """
        N, B, C = feats.shape
        # expand prototype tokens
        proto = self.prototype_tokens.unsqueeze(1).expand(-1, B, -1)  # [P, B, C]
        # attention 聚合
        proto_out, _ = self.attn(proto, feats, feats)  # [P, B, C]
        # proto_out = proto_out.permute(1, 0, 2).contiguous()  # [B, P, C]

        return self.norm(proto_out)

class PrototypeAlignmentPathLearner(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        # 跨注意力: 源原型 Q, 目标原型 K/V
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads,dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(in_dim)

        # 路径优化 MLP (学习对齐embedding)
        self.path_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        self.norm2 = nn.LayerNorm(in_dim)

        # soft alignment score (对齐矩阵)
        self.alignment_score = nn.Linear(in_dim, 1)
        self.path_np = NPPathLearner(out_dim=in_dim, k_neighbors=16)

    def forward(self, src_proto, tgt_proto):
        """
        Args:
            src_proto: [B, P, C] 源点云原型
            tgt_proto: [B, P, C] 目标点云原型
        Returns:
            aligned_src: [B, P, C] 对齐后的源原型
            align_weights: [B, P, P] 源->目标对齐权重
        """
        # 1. 跨注意力
        src_proto=src_proto.permute(1, 0, 2)
        tgt_proto=tgt_proto.permute(1, 0, 2)

        src_aligned, _ = self.cross_attn(src_proto, tgt_proto, tgt_proto)  # [B, P, C]
        src_aligned = self.norm1(src_proto + src_aligned)

        # 2. 路径优化 MLP
        src_path = self.path_mlp(src_aligned)

        # src_alignedxx=src_aligned.permute(1, 0, 2)
        # tgt_protoxx=tgt_proto.permute(1, 0, 2)
        # src_pathxx = self.path_np(src_alignedxx, tgt_protoxx)  # 非参数化路径增强
        # src_pathxx=src_pathxx.permute(1, 0, 2)
        # src_path=src_path+src_pathxx

        src_path = self.norm2(src_aligned + src_path)


        # 3. soft alignment 矩阵 (B, P, P)
        # 用 dot-product 计算相似度

        B, N, C = src_path.shape
        q2=src_path
        k2 = tgt_proto  # 形状 ((B-1)*N, C)
        v2 = k2  # 通常键和值是相同的
        k2 = k2.reshape(-1, C)  # 形状 ((B-1)*N, C)
        v2 = v2.reshape(-1, C)  # 形状 ((B-1)*N, C)
        scale = C ** -0.5  # 缩放因子

        align_weights = torch.softmax(torch.einsum('bnc,mc->bnm', q2, k2) * scale, dim=-1)  # 形状 (1, N, (B-1)*N)
        aligned_src = torch.einsum('bnm,mc->bnc', align_weights, v2)  # 形状 (1, N, C)

        # sim = torch.matmul(src_path, tgt_proto.transpose(-1, -2)) / (src_path.shape[-1] ** 0.5)
        # align_weights = F.softmax(sim, dim=-1)
        #
        # # 4. 得到对齐后的源原型 (加权求和目标原型)
        # aligned_src = torch.matmul(align_weights, tgt_proto)

        return aligned_src, align_weights


class RegTR(GenericRegModel):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)


        #shaoleyige  group

        #######################
        # Preprocessor
        #######################
        self.preprocessor = PreprocessorGPU(cfg)

        #######################
        # KPConv Encoder/decoder
        #######################
        self.kpf_encoder = KPFEncoder(cfg, cfg.d_embed)
        #通过KPConv+残差结构（两侧共享权重），得到关键点（少于原来的点数量）和D维特征（一般取256维）

        # Bottleneck layer to shrink KPConv features to a smaller dimension for running attention
        self.feat_proj = nn.Linear(self.kpf_encoder.encoder_skip_dims[-1], cfg.d_embed, bias=True)

        #######################
        # Embeddings
        #######################
        if cfg.get('pos_emb_type', 'sine') == 'sine':
            self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.d_embed,
                                                         scale=cfg.get('pos_emb_scaling', 1.0))
        elif cfg['pos_emb_type'] == 'learned':
            self.pos_embed = PositionEmbeddingLearned(3, cfg.d_embed)
        else:
            raise NotImplementedError

        #######################
        # Attention propagation
        #######################
        encoder_layer = TransformerCrossEncoderLayer(
            cfg.d_embed, cfg.nhead, cfg.d_feedforward, cfg.dropout,
            activation=cfg.transformer_act,
            normalize_before=cfg.pre_norm,
            sa_val_has_pos_emb=cfg.sa_val_has_pos_emb,
            ca_val_has_pos_emb=cfg.ca_val_has_pos_emb,
            attention_type=cfg.attention_type,
        )
        encoder_norm = nn.LayerNorm(cfg.d_embed) if cfg.pre_norm else None
        self.transformer_encoder = TransformerCrossEncoder(
            encoder_layer, cfg.num_encoder_layers, encoder_norm,
            return_intermediate=True)

        #######################
        # Output layers
        #######################
        if cfg.get('direct_regress_coor', False):
            # self.correspondence_decoder = CorrespondenceRegressor(cfg.d_embed)
            self.correspondence_decoder = CorrespondenceDecoder(cfg.d_embed,
                                                                cfg.corr_decoder_has_pos_emb,
                                                                self.pos_embed)
        else:
            self.correspondence_decoder = CorrespondenceDecoder(cfg.d_embed,
                                                                cfg.corr_decoder_has_pos_emb,
                                                                self.pos_embed)

        #######################
        # Losses
        #######################
        self.overlap_criterion = nn.BCEWithLogitsLoss()
        if self.cfg.feature_loss_type == 'infonce':
            self.feature_criterion = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
        elif self.cfg.feature_loss_type == 'circle':
            self.feature_criterion = CircleLossFull(dist_type='euclidean', r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = self.feature_criterion
        else:
            raise NotImplementedError

        self.corr_criterion = CorrCriterion(metric='mae')

        self.weight_dict = {}
        for k in ['overlap', 'feature', 'corr']:
            for i in cfg.get(f'{k}_loss_on', [cfg.num_encoder_layers - 1]):
                self.weight_dict[f'{k}_{i}'] = cfg.get(f'wt_{k}')
        self.weight_dict['feature_un'] = cfg.wt_feature_un

        self.logger.info('Loss weighting: {}'.format(self.weight_dict))
        self.logger.info(
            f'Config: d_embed:{cfg.d_embed}, nheads:{cfg.nhead}, pre_norm:{cfg.pre_norm}, '
            f'use_pos_emb:{cfg.transformer_encoder_has_pos_emb}, '
            f'sa_val_has_pos_emb:{cfg.sa_val_has_pos_emb}, '
            f'ca_val_has_pos_emb:{cfg.ca_val_has_pos_emb}'
        )

        # 结构语义原型提取器
        self.semantic_proto_extractor = SemanticPrototypeExtractor(
            in_dim=cfg.d_embed,
            num_prototypes=cfg.get("num_prototypes", 8),
            num_heads=cfg.nhead
        )

        # 原型对齐路径优化模块
        self.proto_align_path = PrototypeAlignmentPathLearner(
            in_dim=cfg.d_embed,
            hidden_dim=cfg.get("proto_hidden_dim", 512),
            num_heads=cfg.nhead
    )

        self.alignment_layer = AlignmentDrivenTransformerLayer(
            d_model=cfg.d_embed,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_feedforward,
            dropout=cfg.dropout
        )

    def forward(self, batch):
        B = len(batch['src_xyz'])
        outputs = {}

        if _TIMEIT:
            t_start_all_cuda, t_end_all_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_pp_cuda, t_end_pp_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_all_cuda.record()
            t_start_pp_cuda.record()

        # Preprocess
        kpconv_meta = self.preprocessor(batch['src_xyz'] + batch['tgt_xyz'])
        batch['kpconv_meta'] = kpconv_meta
        slens = [s.tolist() for s in kpconv_meta['stack_lengths']]
        slens_c = slens[-1]
        src_slens_c, tgt_slens_c = slens_c[:B], slens_c[B:]
        feats0 = torch.ones_like(kpconv_meta['points'][0][:, 0:1])

        if _TIMEIT:
            t_end_pp_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_pp_cuda = t_start_pp_cuda.elapsed_time(t_end_pp_cuda) / 1000
            t_start_enc_cuda, t_end_enc_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_enc_cuda.record()

        ####################
        # REGTR Encoder
        ####################
        # KPConv encoder (downsampling) to obtain unconditioned features
        feats_un, skip_x = self.kpf_encoder(feats0, kpconv_meta)
        if _TIMEIT:
            t_end_enc_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_enc_cuda = t_start_enc_cuda.elapsed_time(t_end_enc_cuda) / 1000
            t_start_att_cuda, t_end_att_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_att_cuda.record()

        both_feats_un = self.feat_proj(feats_un)



        src_feats_un, tgt_feats_un = split_src_tgt(both_feats_un, slens_c)

        # Position embedding for downsampled points
        src_xyz_c, tgt_xyz_c = split_src_tgt(kpconv_meta['points'][-1], slens_c) #结果直接进入decode
        src_pe, tgt_pe = split_src_tgt(self.pos_embed(kpconv_meta['points'][-1]), slens_c)
        src_pe_padded, _, _ = pad_sequence(src_pe)
        tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        # Performs padding, then apply attention (REGTR "encoder" stage) to condition on the other
        # point cloud
        src_feats_padded, src_key_padding_mask, _ = pad_sequence(src_feats_un,
                                                                 require_padding_mask=True)
        tgt_feats_padded, tgt_key_padding_mask, _ = pad_sequence(tgt_feats_un,
                                                                 require_padding_mask=True)

        src_feats_cond, tgt_feats_cond = self.transformer_encoder(
            src_feats_padded, tgt_feats_padded,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_pos=src_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
            tgt_pos=tgt_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
        )



        # === 语义原型提取 model1 ===
        # src_feats_last = pad_sequence(unpad_sequences(src_feats_cond[-1], src_slens_c))  # [B, Nmax, C]
        # tgt_feats_last = pad_sequence(unpad_sequences(tgt_feats_cond[-1], tgt_slens_c))  # [B, Mmax, C]
        # src_proto = self.semantic_proto_extractor(src_feats_last)  # [B, P, C]
        # tgt_proto = self.semantic_proto_extractor(tgt_feats_last)  # [B, P, C]


        src_proto = self.semantic_proto_extractor(src_feats_cond[-1])  # 取最后一层 encoder 输出
        # pooled = F.adaptive_avg_pool1d(src_feats_cond[-1].permute(1, 2,0), output_size=src_proto.shape[0])
        # pooled = pooled.permute(2, 0, 1)  # -> [P, B, C] = [128, 4, 256]
        #
        # src_proto=src_proto+pooled

        tgt_proto = self.semantic_proto_extractor(tgt_feats_cond[-1])
        # tgt_proto=tgt_proto+tgt_feats_cond[-1]

        # === 原型对齐路径优化 model2 ===
        aligned_src_proto, align_weights = self.proto_align_path(src_proto, tgt_proto)

        # === 对齐驱动 Transformer model3===
        src_feats_last = src_feats_cond[-1]
        tgt_feats_last = tgt_feats_cond[-1]
        src_feats_last=src_feats_last.permute(1, 0, 2)
        tgt_feats_last=tgt_feats_last.permute(1, 0, 2)

        src_feats_aligned = self.alignment_layer(src_feats_last, tgt_feats_last, aligned_src_proto)
        src_feats_cond = src_feats_cond.clone()
        x=src_feats_cond[:-1]
        src_feats_aligned = src_feats_aligned.permute(1, 0, 2)  # [607, 4, 256]

        src_feats_cond[:-1] = x + src_feats_aligned




        src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list = \
            self.correspondence_decoder(src_feats_cond, tgt_feats_cond, src_xyz_c, tgt_xyz_c)

        src_feats_list = unpad_sequences(src_feats_cond, src_slens_c)
        tgt_feats_list = unpad_sequences(tgt_feats_cond, tgt_slens_c)


        num_pred = src_feats_cond.shape[0]

        ## TIMING CODE
        if _TIMEIT:
            t_end_att_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_att_cuda = t_start_att_cuda.elapsed_time(t_end_att_cuda) / 1000
            t_start_pose_cuda, t_end_pose_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_pose_cuda.record()

        # Stacks correspondences in both directions and computes the pose
        corr_all, overlap_prob = [], []
        for b in range(B):
            corr_all.append(torch.cat([
                torch.cat([src_xyz_c[b].expand(num_pred, -1, -1), src_corr_list[b]], dim=2),
                torch.cat([tgt_corr_list[b], tgt_xyz_c[b].expand(num_pred, -1, -1)], dim=2)
            ], dim=1))
            overlap_prob.append(torch.cat([
                torch.sigmoid(src_overlap_list[b][:, :, 0]),
                torch.sigmoid(tgt_overlap_list[b][:, :, 0]),
            ], dim=1))

            # # Thresholds the overlap probability. Enable this for inference to get a slight boost
            # # in performance. However, we do not use this in the paper.
            # overlap_prob = [nn.functional.threshold(overlap_prob[b], 0.5, 0.0) for b in range(B)]

        pred_pose_weighted = torch.stack([
            compute_rigid_transform(corr_all[b][..., :3], corr_all[b][..., 3:],
                                    overlap_prob[b])
            for b in range(B)], dim=1)

        ## TIMING CODE
        if _TIMEIT:
            t_end_pose_cuda.record()
            t_end_all_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_pose_cuda = t_start_pose_cuda.elapsed_time(t_end_pose_cuda) / 1000
            t_elapsed_all_cuda = t_start_all_cuda.elapsed_time(t_end_all_cuda) / 1000
            with open('timings.txt', 'a') as fid:
                fid.write('{:10f}\t{:10f}\t{:10f}\t{:10f}\t{:10f}\n'.format(
                    t_elapsed_pp_cuda, t_elapsed_enc_cuda, t_elapsed_att_cuda,
                    t_elapsed_pose_cuda, t_elapsed_all_cuda
                ))

        outputs = {
            # Predictions
            'src_feat_un': src_feats_un,
            'tgt_feat_un': tgt_feats_un,
            'src_feat': src_feats_list,  # List(B) of (N_pred, N_src, D)
            'tgt_feat': tgt_feats_list,  # List(B) of (N_pred, N_tgt, D)

            'src_kp': src_xyz_c,
            'src_kp_warped': src_corr_list,
            'tgt_kp': tgt_xyz_c,
            'tgt_kp_warped': tgt_corr_list,

            'src_overlap': src_overlap_list,
            'tgt_overlap': tgt_overlap_list,

            'pose': pred_pose_weighted,
        }
        return outputs

    def compute_loss(self, pred, batch,batch_idx):
        losses = {}
        kpconv_meta = batch['kpconv_meta']
        pose_gt = batch['pose']
        p = len(kpconv_meta['stack_lengths']) - 1  # coarsest level

        # Compute groundtruth overlaps first
        batch['overlap_pyr'] = compute_overlaps(batch)
        src_overlap_p, tgt_overlap_p = \
            split_src_tgt(batch['overlap_pyr'][f'pyr_{p}'], kpconv_meta['stack_lengths'][p])

        # Overlap prediction loss
        all_overlap_pred = torch.cat(pred['src_overlap'] + pred['tgt_overlap'], dim=-2)
        all_overlap_gt = batch['overlap_pyr'][f'pyr_{p}']
        for i in self.cfg.overlap_loss_on:
            losses[f'overlap_{i}'] = self.overlap_criterion(all_overlap_pred[i, :, 0], all_overlap_gt)

        # Feature criterion
        for i in self.cfg.feature_loss_on:
            losses[f'feature_{i}'] = self.feature_criterion(
                [s[i] for s in pred['src_feat']],
                [t[i] for t in pred['tgt_feat']],
                se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
            )
        losses['feature_un'] = self.feature_criterion_un(
            pred['src_feat_un'],
            pred['tgt_feat_un'],
            se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
        )

        # Loss on the 6D correspondences
        for i in self.cfg.corr_loss_on:
            src_corr_loss = self.corr_criterion(
                pred['src_kp'],
                [w[i] for w in pred['src_kp_warped']],
                batch['pose'],
                overlap_weights=src_overlap_p
            )
            tgt_corr_loss = self.corr_criterion(
                pred['tgt_kp'],
                [w[i] for w in pred['tgt_kp_warped']],
                torch.stack([se3_inv(p) for p in batch['pose']]),
                overlap_weights=tgt_overlap_p
            )
            losses[f'corr_{i}'] = src_corr_loss + tgt_corr_loss

        debug = False  # Set this to true to look at the registration result
        # if debug:
#二维图
        b = 0
        o = -1
        out = visualize_registration_offline(batch['src_xyz'][b], batch['tgt_xyz'][b],
                               torch.cat([pred['src_kp'][b], pred['src_kp_warped'][b][o]], dim=1),
                               correspondence_conf=torch.sigmoid(pred['src_overlap'][b][o])[:, 0],
                                             pose_gt=pose_gt[b], pose_pred=pred['pose'][o, b],
        out_path=f'./results/oursmodello/vis_sample_{batch_idx}.png')
        print('Saved visualization to', out)




        b = 0
        o = -1  # Visualize output of final transformer layer
        pose_gt_b = to_homogeneous(to_numpy(pose_gt[b]))
        pose_pred_b = to_homogeneous(to_numpy(pred['pose'][o, b]))


        save_registration_mat(batch['src_xyz'][b], batch['tgt_xyz'][b],
                               torch.cat([pred['src_kp'][b], pred['src_kp_warped'][b][o]], dim=1),
                               correspondence_conf=torch.sigmoid(pred['src_overlap'][b][o])[:, 0],
                              pose_gt=pose_gt_b,
                              pose_pred=pose_pred_b,
                              pose_index=None,
                              out_mat=f'./results/oursmodello/vis_sample_{batch_idx}.mat'
                              )

        losses['total'] = torch.sum(
            torch.stack([(losses[k] * self.weight_dict[k]) for k in losses]))
        return losses

import numpy as np
def to_homogeneous(Rt):
    """
    Rt: (3,4) -> 4x4 homogeneous
    """
    if Rt.shape == (3,4):
        T = np.eye(4, dtype=np.float32)
        T[:3,:4] = Rt
        return T
    elif Rt.shape == (4,4):
        return Rt
    else:
        raise ValueError(f"Unsupported pose shape {Rt.shape}")

def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)



class CorrespondenceDecoder(nn.Module):
    def __init__(self, d_embed, use_pos_emb, pos_embed=None, num_neighbors=0, use_geom_bias=True):
        super().__init__()

        assert use_pos_emb is False or pos_embed is not None, \
            'Position encoder must be supplied if use_pos_emb is True'

        self.use_pos_emb = use_pos_emb
        self.pos_embed = pos_embed
        self.q_norm = nn.LayerNorm(d_embed)

        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_embed, d_embed)
        self.conf_logits_decoder = nn.Linear(d_embed, 1)
        self.num_neighbors = num_neighbors

        self.geo_enc = nn.Sequential(
            nn.Linear(3, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed)
        )
        self.proj_coord = nn.Linear(d_embed, 3)



        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)

    def simple_attention(self, query, key, value, src_xyz, tgt_xyz, key_padding_mask=None):
        """Simplified single-head attention that does not project the value:
        Linearly projects only the query and key, compute softmax dot product
        attention, then returns the weighted sum of the values

        Args:
            query: ([N_pred,] Q, B, D)
            key: ([N_pred,] S, B, D)
            value: (S, B, E), i.e. dimensionality can be different
            key_padding_mask: (B, S)

        Returns:
            Weighted values (B, Q, E)
        """

        q = self.q_proj(query) / math.sqrt(query.shape[-1])
        k = self.k_proj(key)

        attn = torch.einsum('...qbd,...sbd->...bqs', q, k)

        src_xyz_b = src_xyz.permute(1, 0, 2).unsqueeze(0).expand(q.shape[0], -1, -1, -1)  # (N_pred, B, Q, 3)
        tgt_xyz_b = tgt_xyz.permute(1, 0, 2).unsqueeze(0).expand(q.shape[0], -1, -1, -1)  # (N_pred, B, S, 3)

        dist = torch.cdist(src_xyz_b, tgt_xyz_b)  # (N_pred, B, Q, S)
        mean_dist = dist.mean(dim=-1, keepdim=True) + 1e-6  # (N_pred, B, Nq, 1)：每个源点的平均距离（归一化）
        geom_bias = -dist / mean_dist  # (N_pred, B, Nq, Nk)：距离越大，Bias越小（抑制远距点）

        attn = attn + geom_bias


        if key_padding_mask is not None:
            attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(key_padding_mask, float('-inf'))
            attn = attn + attn_mask[:, None, :]  # ([N_pred,] B, Q, S)

        if self.num_neighbors > 0:
            neighbor_mask = torch.full_like(attn, fill_value=float('-inf'))
            haha = torch.topk(attn, k=self.num_neighbors, dim=-1).indices
            neighbor_mask[:, :, haha] = 0
            attn = attn + neighbor_mask

        attn = torch.softmax(attn, dim=-1)

        attn_out = torch.einsum('...bqs,...sbd->...qbd', attn, value)

        return attn_out

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)6 607 4 256
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)6 592 4 256
            src_xyz: List of ([N_pred,] N_src, 3)
            tgt_xyz: List of ([N_pred,] N_tgt, 3)

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)
        assert src_xyz_padded.shape[:-1] == src_feats_padded.shape[-3:-1] and \
               tgt_xyz_padded.shape[:-1] == tgt_feats_padded.shape[-3:-1]

        if self.use_pos_emb:
            both_xyz_packed = torch.cat(src_xyz + tgt_xyz)
            slens = list(map(len, src_xyz)) + list(map(len, tgt_xyz))
            src_pe, tgt_pe = split_src_tgt(self.pos_embed(both_xyz_packed), slens)
            src_pe_padded, _, _ = pad_sequence(src_pe)
            tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        src_geo_feat = self.geo_enc(src_xyz_padded)  # (N_pred, N_src_pad, B, d_embed)
        tgt_geo_feat = self.geo_enc(tgt_xyz_padded)  # (N_pred, N_tgt_pad, B, d_embed)


        # Decode the coordinates
        src_feats2 = src_feats_padded + src_pe_padded +src_geo_feat if self.use_pos_emb else src_feats_padded
        tgt_feats2 = tgt_feats_padded + tgt_pe_padded +tgt_geo_feat if self.use_pos_emb else tgt_feats_padded
        src_corr = self.simple_attention(src_feats2, tgt_feats2, pad_sequence(tgt_xyz)[0],src_xyz_padded, tgt_xyz_padded,
                                         tgt_key_padding_mask)
        tgt_corr = self.simple_attention(tgt_feats2, src_feats2, pad_sequence(src_xyz)[0],tgt_xyz_padded, src_xyz_padded,
                                         src_key_padding_mask)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list


class CorrespondenceRegressor(nn.Module):

    def __init__(self, d_embed):
        super().__init__()

        self.coor_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 3)
        )
        self.conf_logits_decoder = nn.Linear(d_embed, 1)

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3). Ignored
            tgt_xyz: List of ([N_pred,] N_tgt, 3). Ignored

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)

        # Decode the coordinates
        src_corr = self.coor_mlp(src_feats_padded)
        tgt_corr = self.coor_mlp(tgt_feats_padded)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list
