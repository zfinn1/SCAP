"""Modified from DETR's transformer.py

- Cross encoder layer is similar to the decoder layers in Transformer, but
  updates both source and target features
- Added argument to control whether value has position embedding or not for
  TransformerEncoderLayer and TransformerDecoderLayer
- Decoder layer now keeps track of attention weights
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerCrossEncoder(nn.Module):

    def __init__(self, cross_encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        src_intermediate, tgt_intermediate = [], []

        for layer in self.layers:
            src, tgt = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos, tgt_pos=tgt_pos)
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        if self.norm is not None:
            src = self.norm(src)
            tgt = self.norm(tgt)
            if self.return_intermediate:
                if len(self.layers) > 0:
                    src_intermediate.pop()
                    tgt_intermediate.pop()
                src_intermediate.append(src)
                tgt_intermediate.append(tgt)

        if self.return_intermediate:
            return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src.unsqueeze(0), tgt.unsqueeze(0)

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError
        #改成了PTA


        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis


        #少了3 4 5 层
        #一些feature_scatter 相关操作 3 个
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)#  多出来的

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        src2 = self.norm1(src)
        src2_w_pos = self.with_pos_embed(src2, src_pos)
        q = k = src2_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                                              value=src2_w_pos if self.sa_val_has_pos_emb else src2,
                                              attn_mask=src_mask,
                                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        tgt2 = self.norm1(tgt)
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)
        q = k = tgt2_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt3, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src2,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class TransformerCrossEncoderLayer2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError
        #改成了PTA


        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

        self.qkv = nn.Linear(256, 256 * 3, bias=False)
        self.proj = nn.Linear(256, 256)
        self.proj_drop = nn.Dropout(0.1)
        #少了3 4 5 层
        #一些feature_scatter 相关操作 3 个
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)#  多出来的

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        src2 = self.norm1(src)#633 4 256  [N, B, C]
        src2_w_pos = self.with_pos_embed(src2, src_pos)#633 4 256

        x = src2_w_pos.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 8, C // 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.
        k = F.elu(k) + 1.

        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            q = q * mask
            k = k * mask
            v = v * mask

        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, h, d, d]
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)  # [B, h, N]
        x = torch.einsum('bhde,bhnd,bhn->bhne', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = x.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)

        x = self.proj(x)
        # x = self.proj_drop(x)
        src = src + self.dropout1(x)

        # q = k = src2_w_pos
        # src2, satt_weights_s = self.self_attn(q, k,
        #                                       value=src2_w_pos if self.sa_val_has_pos_emb else src2,
        #                                       attn_mask=src_mask,
        #                                       key_padding_mask=src_key_padding_mask)








        tgt2 = self.norm1(tgt)#628 4 256
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        x = tgt2_w_pos.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 8, C // 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.
        k = F.elu(k) + 1.

        if tgt_key_padding_mask is not None:
            mask = (~tgt_key_padding_mask).float().unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            q = q * mask
            k = k * mask
            v = v * mask

        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, h, d, d]
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)  # [B, h, N]
        x = torch.einsum('bhde,bhnd,bhn->bhne', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = x.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)

        x = self.proj(x)


        tgt = tgt + self.dropout1(x)

        # q = k = tgt2_w_pos
        # tgt2, satt_weights_t = self.self_attn(q, k,
        #                                       value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
        #                                       attn_mask=tgt_mask,
        #                                       key_padding_mask=tgt_key_padding_mask)




        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt3, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src2,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        # self.satt_weights = (satt_weights_s, satt_weights_t)
        # self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)



class TransformerCrossEncoderLayer3(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError
        #改成了PTA


        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

        self.qkv = nn.Linear(256, 256 * 3, bias=False)
        self.proj = nn.Linear(256, 256)
        self.proj_drop = nn.Dropout(0.1)
        #少了3 4 5 层
        #一些feature_scatter 相关操作 3 个
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)#  多出来的

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        src2 = self.norm1(src)#633 4 256  [N, B, C]
        src2_w_pos = self.with_pos_embed(src2, src_pos)#633 4 256

        x = src2_w_pos.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 8, C // 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.
        k = F.elu(k) + 1.

        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            q = q * mask
            k = k * mask
            v = v * mask

        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, h, d, d]
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)  # [B, h, N]
        x = torch.einsum('bhde,bhnd,bhn->bhne', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = x.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)

        x = self.proj(x)
        # x = self.proj_drop(x)
        src = src + self.dropout1(x)

        # q = k = src2_w_pos
        # src2, satt_weights_s = self.self_attn(q, k,
        #                                       value=src2_w_pos if self.sa_val_has_pos_emb else src2,
        #                                       attn_mask=src_mask,
        #                                       key_padding_mask=src_key_padding_mask)








        tgt2 = self.norm1(tgt)#628 4 256
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        x = tgt2_w_pos.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 8, C // 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.
        k = F.elu(k) + 1.

        if tgt_key_padding_mask is not None:
            mask = (~tgt_key_padding_mask).float().unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            q = q * mask
            k = k * mask
            v = v * mask

        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, h, d, d]
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)  # [B, h, N]
        x = torch.einsum('bhde,bhnd,bhn->bhne', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = x.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)

        x = self.proj(x)


        tgt = tgt + self.dropout1(x)

        # q = k = tgt2_w_pos
        # tgt2, satt_weights_t = self.self_attn(q, k,
        #                                       value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
        #                                       attn_mask=tgt_mask,
        #                                       key_padding_mask=tgt_key_padding_mask)




        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)# 607 4  256
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)# 592 4 256

        src_w_pos= src_w_pos.permute(1, 0, 2)
        tgt_w_pos= tgt_w_pos.permute(1, 0, 2)

        #src
        q2=src_w_pos
        k2=tgt_w_pos
        # v2=k2
        v2 = k2  # 通常键和值是相同的
        B, N, C = q2.shape

        # k2 = k2.reshape(-1, C)  # 形状 ((B-1)*N, C)
        # v2 = v2.reshape(-1, C)  # 形状 ((B-1)*N, C)
        scale = C ** -0.5  # 缩放因子
        attn_logits = torch.einsum('bnc,bmc->bnm', q2, k2) * scale
        if tgt_key_padding_mask is not None:
            # mask: [B, 1, M] -> broadcast到 N
            mask = (~tgt_key_padding_mask).unsqueeze(1).to(attn_logits.dtype)  # 1=有效, 0=pad
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, N, M]
        src3 = torch.einsum('bnm,bmc->bnc', attn_weights, v2)
        src3 = src3.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)

        # src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
        #                                            key=tgt_w_pos,
        #                                            value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
        #                                            attn_mask=tgt_mask,
        #                                            key_padding_mask=tgt_key_padding_mask)

        #tgt3
        q2=tgt_w_pos
        k2=src_w_pos
        # v2=k2
        v2 = k2  # 通常键和值是相同的
        B, N, C = q2.shape

        # k2 = k2.reshape(-1, C)  # 形状 ((B-1)*N, C)
        # v2 = v2.reshape(-1, C)  # 形状 ((B-1)*N, C)
        scale = C ** -0.5  # 缩放因子
        attn_logits = torch.einsum('bnc,bmc->bnm', q2, k2) * scale
        if src_key_padding_mask is not None:
            # mask: [B, 1, M] -> broadcast到 N
            mask = (~src_key_padding_mask).unsqueeze(1).to(attn_logits.dtype)  # 1=有效, 0=pad
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, N, M]
        tgt3 = torch.einsum('bnm,bmc->bnc', attn_weights, v2)
        tgt3 = tgt3.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)

        #
        # tgt3, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
        #                                            key=src_w_pos,
        #                                            value=src_w_pos if self.ca_val_has_pos_emb else src2,
        #                                            attn_mask=src_mask,
        #                                            key_padding_mask=src_key_padding_mask)

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        # self.satt_weights = (satt_weights_s, satt_weights_t)
        # self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)



class TransformerCrossEncoderLayer4(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError
        #改成了PTA


        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis


        #少了3 4 5 层
        #一些feature_scatter 相关操作 3 个
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)#  多出来的

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        src2 = self.norm1(src)
        src2_w_pos = self.with_pos_embed(src2, src_pos)
        x = src2_w_pos.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 8, C // 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.
        k = F.elu(k) + 1.

        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            q = q * mask
            k = k * mask
            v = v * mask

        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, h, d, d]
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)  # [B, h, N]
        x = torch.einsum('bhde,bhnd,bhn->bhne', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = x.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)

        x = self.proj(x)
        # x = self.proj_drop(x)
        src = src + self.dropout1(x)

        # q = k = src2_w_pos
        # src2, satt_weights_s = self.self_attn(q, k,
        #                                       value=src2_w_pos if self.sa_val_has_pos_emb else src2,
        #                                       attn_mask=src_mask,
        #                                       key_padding_mask=src_key_padding_mask)

        tgt2 = self.norm1(tgt)  # 628 4 256
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        x = tgt2_w_pos.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 8, C // 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.
        k = F.elu(k) + 1.

        if tgt_key_padding_mask is not None:
            mask = (~tgt_key_padding_mask).float().unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            q = q * mask
            k = k * mask
            v = v * mask

        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, h, d, d]
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)  # [B, h, N]
        x = torch.einsum('bhde,bhnd,bhn->bhne', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = x.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)

        x = self.proj(x)

        tgt = tgt + self.dropout1(x)

        # q = k = tgt2_w_pos
        # tgt2, satt_weights_t = self.self_attn(q, k,
        #                                       value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
        #                                       attn_mask=tgt_mask,
        #                                       key_padding_mask=tgt_key_padding_mask)

        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)  # 607 4  256
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)  # 592 4 256

        src_w_pos = src_w_pos.permute(1, 0, 2)
        tgt_w_pos = tgt_w_pos.permute(1, 0, 2)

        # src
        q2 = src_w_pos
        k2 = tgt_w_pos
        # v2=k2
        v2 = k2  # 通常键和值是相同的
        B, N, C = q2.shape

        # k2 = k2.reshape(-1, C)  # 形状 ((B-1)*N, C)
        # v2 = v2.reshape(-1, C)  # 形状 ((B-1)*N, C)
        scale = C ** -0.5  # 缩放因子
        attn_logits = torch.einsum('bnc,bmc->bnm', q2, k2) * scale
        if tgt_key_padding_mask is not None:
            # mask: [B, 1, M] -> broadcast到 N
            mask = (~tgt_key_padding_mask).unsqueeze(1).to(attn_logits.dtype)  # 1=有效, 0=pad
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, N, M]
        src3 = torch.einsum('bnm,bmc->bnc', attn_weights, v2)
        src3 = src3.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)

        # src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
        #                                            key=tgt_w_pos,
        #                                            value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
        #                                            attn_mask=tgt_mask,
        #                                            key_padding_mask=tgt_key_padding_mask)

        # tgt3
        q2 = tgt_w_pos
        k2 = src_w_pos
        # v2=k2
        v2 = k2  # 通常键和值是相同的
        B, N, C = q2.shape

        # k2 = k2.reshape(-1, C)  # 形状 ((B-1)*N, C)
        # v2 = v2.reshape(-1, C)  # 形状 ((B-1)*N, C)
        scale = C ** -0.5  # 缩放因子
        attn_logits = torch.einsum('bnc,bmc->bnm', q2, k2) * scale
        if src_key_padding_mask is not None:
            # mask: [B, 1, M] -> broadcast到 N
            mask = (~src_key_padding_mask).unsqueeze(1).to(attn_logits.dtype)  # 1=有效, 0=pad
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, N, M]
        tgt3 = torch.einsum('bnm,bmc->bnc', attn_weights, v2)
        tgt3 = tgt3.permute(1, 0, 2)  # 转置维度 (2, 0, 1) → (B=4, N=633, C=256)

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
