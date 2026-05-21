# Put this in your utils/viz.py or similar and import
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互，用于无头服务器
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os
import torch
import math

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)

def _maybe_subsample(pts, max_pts=3000, seed=0):
    """If pts too many, subsample for plotting speed"""
    n = pts.shape[0]
    if n <= max_pts:
        return pts
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_pts, replace=False)
    return pts[idx]

def visualize_registration_offline(src_xyz, tgt_xyz, correspondences,
                                   correspondence_conf=None,
                                   pose_gt=None, pose_pred=None,
                                   out_path='vis_registration.png',
                                   subsample_pts=3000):
    """
    非交互式可视化：生成 2x3 的 PNG，保存在 out_path
    Args:
        src_xyz, tgt_xyz: (N,3) arrays or tensors (原始点云)
        correspondences: (K,6) array/tensor: [src_xyz, tgt_xyz] 或者包含N个对应对
        correspondence_conf: (K,) optional confidence values (0..1)
        pose_gt / pose_pred: optional 4x4 transforms (torch or np) to apply to src for "aligned views"
        out_path: 保存路径
    """
    src = to_numpy(src_xyz)
    tgt = to_numpy(tgt_xyz)
    corr = to_numpy(correspondences)

    # 防止太大影响绘图速度
    src_plot = _maybe_subsample(src, max_pts=subsample_pts, seed=1)
    tgt_plot = _maybe_subsample(tgt, max_pts=subsample_pts, seed=2)

    # prepare correspondences lines (use full correspondences but limit number)
    if corr is None or corr.shape[0] == 0:
        corr = np.zeros((0,6))
    # choose a subset of correspondences for drawing lines if too many
    K = corr.shape[0]
    max_lines = 500
    line_idx = np.arange(K) if K <= max_lines else np.random.choice(K, max_lines, replace=False)
    corr_lines = corr[line_idx]

    # transform src by GT if provided for some panels
    def apply_se3(T, pts):
        if T is None:
            return pts
        T = to_numpy(T)
        R = T[:3,:3]
        t = T[:3,3]
        return (R @ pts.T).T + t

    src_gt = apply_se3(pose_gt, src)
    src_pred = apply_se3(pose_pred, src)

    # confidence colormap for keypoint sets (if provided)
    if correspondence_conf is not None:
        conf = to_numpy(correspondence_conf)
        # normalize to [0,1]
        cmin, cmax = conf.min(), conf.max()
        if cmax - cmin < 1e-8:
            conf_norm = np.clip(conf, 0, 1)
        else:
            conf_norm = (conf - cmin) / (cmax - cmin)
        # subset
        conf_lines = conf[line_idx]
        cmap = plt.get_cmap('autumn')
        src_colors = cmap(conf_norm)[:,:3]
        src_colors_lines = cmap(conf_lines)[:,:3]
        tgt_colors = plt.get_cmap('summer')(conf_norm)[:,:3]
        tgt_colors_lines = plt.get_cmap('summer')(conf_lines)[:,:3]
    else:
        src_colors = None
        tgt_colors = None
        src_colors_lines = None
        tgt_colors_lines = None

    # Build 6 subplots (3D)
    fig = plt.figure(figsize=(18.5, 12))
    axs = [fig.add_subplot(2, 3, i+1, projection='3d') for i in range(6)]

    def set_ax(ax, title):
        ax.set_title(title)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        # keep equal aspect (approx)
        ax.set_box_aspect((1,1,1))

    # 0: Source only (with source keypoints = correspondences src positions)
    set_ax(axs[0], '0: Source (keypoints)')
    axs[0].scatter(src_plot[:,0], src_plot[:,1], src_plot[:,2], s=1, c='red', alpha=0.6)
    if corr.shape[0] > 0:
        src_kps = corr[:,:3]
        axs[0].scatter(src_kps[:,0], src_kps[:,1], src_kps[:,2], s=8, c='darkred')

    # 1: Source + Target with lines indicating src keypoints -> their transformed/predicted locations (corr)
    set_ax(axs[1], '1: src->tgt correspondences')
    axs[1].scatter(src_plot[:,0], src_plot[:,1], src_plot[:,2], s=1, c='red', alpha=0.4)
    axs[1].scatter(tgt_plot[:,0], tgt_plot[:,1], tgt_plot[:,2], s=1, c='green', alpha=0.4)
    if corr_lines.shape[0] > 0:
        segments = np.stack([corr_lines[:,:3], corr_lines[:,3:]], axis=1)  # (L,2,3)
        segments = segments.reshape(-1,2,3)
        lc = Line3DCollection(segments, linewidths=0.5, colors='gray', alpha=0.6)
        axs[1].add_collection3d(lc)

    # 2: Source and target under GT alignment (without clutter)
    set_ax(axs[2], '2: GT alignment')
    if src_gt is not None:
        src_gt_plot = _maybe_subsample(src_gt, max_pts=subsample_pts, seed=3)
        axs[2].scatter(src_gt_plot[:,0], src_gt_plot[:,1], src_gt_plot[:,2], s=1, c='red')
    axs[2].scatter(tgt_plot[:,0], tgt_plot[:,1], tgt_plot[:,2], s=1, c='green')

    # 3: Target only with predicted transformed source keypoints
    set_ax(axs[3], '3: Target + pred src keypoints')
    axs[3].scatter(tgt_plot[:,0], tgt_plot[:,1], tgt_plot[:,2], s=1, c='green', alpha=0.6)
    if corr.shape[0] > 0:
        tgt_kps = corr[:,3:]
        axs[3].scatter(tgt_kps[line_idx,0], tgt_kps[line_idx,1], tgt_kps[line_idx,2], s=8, c='blue')

    # 4: GT overlay with keypoints+lines
    set_ax(axs[4], '4: GT overlay (keypoints + lines)')
    if src_gt is not None:
        src_gt_plot2 = _maybe_subsample(src_gt, max_pts=subsample_pts, seed=4)
        axs[4].scatter(src_gt_plot2[:,0], src_gt_plot2[:,1], src_gt_plot2[:,2], s=1, c='red', alpha=0.6)
    axs[4].scatter(tgt_plot[:,0], tgt_plot[:,1], tgt_plot[:,2], s=1, c='green', alpha=0.6)
    if corr_lines.shape[0] > 0:
        segments = np.stack([corr_lines[:,:3], corr_lines[:,3:]], axis=1)
        lc2 = Line3DCollection(segments.reshape(-1,2,3), colors='gray', linewidths=0.5, alpha=0.8)
        axs[4].add_collection3d(lc2)

    # 5: Predicted alignment (without clutter)
    set_ax(axs[5], '5: Pred alignment')
    if src_pred is not None:
        src_pred_plot = _maybe_subsample(src_pred, max_pts=subsample_pts, seed=5)
        axs[5].scatter(src_pred_plot[:,0], src_pred_plot[:,1], src_pred_plot[:,2], s=1, c='red')
    axs[5].scatter(tgt_plot[:,0], tgt_plot[:,1], tgt_plot[:,2], s=1, c='green')

    # Autoscale each axis
    for ax in axs:
        try:
            ax.auto_scale_xyz([np.min(np.concatenate([src[:,0], tgt[:,0]])), np.max(np.concatenate([src[:,0], tgt[:,0]]))],
                              [np.min(np.concatenate([src[:,1], tgt[:,1]])), np.max(np.concatenate([src[:,1], tgt[:,1]]))],
                              [np.min(np.concatenate([src[:,2], tgt[:,2]])), np.max(np.concatenate([src[:,2], tgt[:,2]]))])
        except Exception:
            pass

    plt.tight_layout()
    # ensure output directory exists
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path
