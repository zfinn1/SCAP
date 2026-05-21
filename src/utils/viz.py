import numpy as np
import torch
from matplotlib.pyplot import cm as colormap

import cvhelpers.visualization as cvv
import cvhelpers.colors as colors
from cvhelpers.torch_helpers import to_numpy
from utils.se3_torch import se3_transform


import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互，用于无头服务器
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os
import torch
import math



# save_vis_to_mat.py
import numpy as np
import torch
import os
import scipy.io as sio
import matplotlib.pyplot as plt

def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)

def apply_se3(T, pts):
    """T: 4x4 numpy or torch (or None). pts: (N,3) numpy"""
    if T is None:
        return pts
    T = to_numpy(T)
    if T.shape == (4,4):
        R = T[:3, :3]
        t = T[:3, 3]
        return (R @ pts.T).T + t
    else:
        raise ValueError("pose must be 4x4 transform")

def _maybe_subsample(pts, max_pts=3000, seed=0):
    if pts is None:
        return None
    n = pts.shape[0]
    if n <= max_pts:
        return pts.copy()
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_pts, replace=False)
    return pts[idx]

def _colormap_rgb(conf, cmap_name='autumn'):
    """conf: (K,) in [0,1] or arbitrary -> return (K,3) float rgb [0..1]"""
    cmap = plt.get_cmap(cmap_name)
    conf = np.array(conf)
    if conf.size == 0:
        return np.zeros((0,3), dtype=float)
    # normalize to [0,1]
    if conf.max() - conf.min() < 1e-9:
        conf_norm = np.clip(conf, 0.0, 1.0)
    else:
        conf_norm = (conf - conf.min()) / (conf.max() - conf.min())
    rgb = cmap(conf_norm)[:, :3]
    return rgb.astype(np.float32)

def save_registration_mat(src_xyz, tgt_xyz, correspondences,
                          correspondence_conf=None,
                          pose_gt=None, pose_pred=None,
                          out_mat='./vis6.mat',
                          subsample_pts=3000,
                          max_lines=500):
    """
    保存可用于 MATLAB 恢复 6 个视图的数据到 out_mat 文件。
    输入:
        src_xyz: (N,3) array or torch tensor
        tgt_xyz: (M,3) array or torch tensor
        correspondences: (K,6) array/tensor where each row [sx,sy,sz, tx,ty,tz]
        correspondence_conf: (K,) optional confidences (float)
        pose_gt: 4x4 transform (numpy or tensor) or None
        pose_pred: 4x4 transform or None
        out_mat: 保存路径 (.mat)
    输出:
        保存多个变量到 out_mat，变量名为：
          src_full, tgt_full, corr_full, corr_conf,
          p0_src_plot, p0_src_kps, ...
          p1_src_plot, p1_tgt_plot, p1_lines, p1_line_idx, p1_line_colors, ...
          axis_lims (3x2 matrix: [xmin xmax; ymin ymax; zmin zmax])
    """
    # convert to numpy
    src = to_numpy(src_xyz).astype(np.float32)
    tgt = to_numpy(tgt_xyz).astype(np.float32)
    corr = to_numpy(correspondences).astype(np.float32) if correspondences is not None else np.zeros((0,6), dtype=np.float32)
    conf = to_numpy(correspondence_conf).astype(np.float32) if correspondence_conf is not None else np.zeros((corr.shape[0],), dtype=np.float32)

    # compute transformed src (GT/pred)
    src_gt = apply_se3(pose_gt, src) if src is not None else None
    src_pred = apply_se3(pose_pred, src) if src is not None else None

    # plotting subsamples
    src_plot = _maybe_subsample(src, max_pts=subsample_pts, seed=1)
    tgt_plot = _maybe_subsample(tgt, max_pts=subsample_pts, seed=2)
    src_gt_plot = _maybe_subsample(src_gt, max_pts=subsample_pts, seed=3) if src_gt is not None else None
    src_pred_plot = _maybe_subsample(src_pred, max_pts=subsample_pts, seed=5) if src_pred is not None else None

    # prepare line subset indices (for panels that draw lines)
    K = corr.shape[0]
    if K == 0:
        line_idx = np.zeros((0,), dtype=np.int32)
    else:
        line_idx = np.arange(K) if K <= max_lines else np.random.choice(K, size=max_lines, replace=False)
    corr_lines = corr[line_idx] if K>0 else np.zeros((0,6), dtype=np.float32)
    conf_lines = conf[line_idx] if (conf is not None and conf.size>0) else np.zeros((corr_lines.shape[0],), dtype=np.float32)

    # colors for keypoints based on conf (store mapped RGB [0..1])
    src_kp_colors = _colormap_rgb(conf, 'autumn') if conf.size>0 else np.zeros((corr.shape[0],3), dtype=np.float32)
    tgt_kp_colors = _colormap_rgb(conf, 'summer') if conf.size>0 else np.zeros((corr.shape[0],3), dtype=np.float32)
    line_colors = _colormap_rgb(conf_lines, 'gray') if conf_lines.size>0 else np.tile(np.array([[0.5,0.5,0.5]], dtype=np.float32), (corr_lines.shape[0],1))

    # compute axis limits (global bounding box for consistent view)
    all_pts = np.concatenate([src.reshape(-1,3), tgt.reshape(-1,3)], axis=0)
    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)
    axis_lims = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]], dtype=np.float32)  # 3x2

    # prepare matdict with clear variable names per panel (p0..p5)
    matdict = {}
    # store full data
    matdict['src_full'] = src
    matdict['tgt_full'] = tgt
    matdict['corr_full'] = corr
    matdict['corr_conf'] = conf
    matdict['src_gt_full'] = src_gt if src_gt is not None else np.zeros((0,3),dtype=np.float32)
    matdict['src_pred_full'] = src_pred if src_pred is not None else np.zeros((0,3),dtype=np.float32)

    # axis limits
    matdict['axis_lims'] = axis_lims

    # Panel 0: Source only (with source keypoints = correspondences src positions)
    matdict['p0_src_plot'] = src_plot
    matdict['p0_src_kps'] = corr[:,:3]  # all src keypoints from correspondences

    # Panel 1: Source + Target with lines (src->tgt)
    matdict['p1_src_plot'] = src_plot
    matdict['p1_tgt_plot'] = tgt_plot
    matdict['p1_lines'] = corr_lines  # L x 6 rows [sx,sy,sz, tx,ty,tz]
    matdict['p1_line_idx'] = line_idx
    matdict['p1_line_colors'] = line_colors

    # Panel 2: GT alignment (src transformed by GT) and target
    matdict['p2_src_gt_plot'] = src_gt_plot if src_gt_plot is not None else np.zeros((0,3),dtype=np.float32)
    matdict['p2_tgt_plot'] = tgt_plot

    # Panel 3: Target only with predicted transformed source keypoints (tgt + predicted tgt keypoints)
    # predicted target keypoints for correspondences are correspondences[:, 3:]
    matdict['p3_tgt_plot'] = tgt_plot
    matdict['p3_tgt_kps'] = corr[:, 3:]

    # Panel 4: GT overlay with keypoints + lines
    matdict['p4_src_gt_plot'] = src_gt_plot if src_gt_plot is not None else np.zeros((0,3),dtype=np.float32)
    matdict['p4_tgt_plot'] = tgt_plot
    matdict['p4_src_kps'] = corr[:,:3]
    matdict['p4_tgt_kps'] = corr[:,3:]
    matdict['p4_lines'] = corr_lines
    matdict['p4_line_colors'] = line_colors

    # Panel 5: Predicted alignment (src transformed by predicted pose) vs target
    matdict['p5_src_pred_plot'] = src_pred_plot if src_pred_plot is not None else np.zeros((0,3),dtype=np.float32)
    matdict['p5_tgt_plot'] = tgt_plot

    # Save also kp colors (full length) so we can color each kp in MATLAB
    matdict['src_kp_colors_full'] = src_kp_colors
    matdict['tgt_kp_colors_full'] = tgt_kp_colors
    matdict['p1_conf_lines'] = conf_lines  # confidences for plotted lines
    matdict['corr_lines_idx'] = line_idx

    # final: write to .mat
    os.makedirs(os.path.dirname(out_mat) or '.', exist_ok=True)
    sio.savemat(out_mat, matdict, do_compression=True)
    print(f"[save_registration_mat] saved to {out_mat} with keys: {list(matdict.keys())}")
    return out_mat



















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


def visualize_registration(src_xyz, tgt_xyz, correspondences,
                           correspondence_conf=None,
                           pose_gt=None, pose_pred=None):
    """Visualize registration, shown as a 2x3 grid:

    -------------
    | 0 | 1 | 2 |
    -------------
    | 3 | 4 | 5 |
    -------------

    0: Source point cloud with source keypoints
    1: Source and target point clouds, with lines indicating source keypoints to
       their transformed locations
    2: Source and target point clouds under groundtruth alignment (without clutter)
    3: Target point cloud with predicted transformed source keypoints
    4: Source and target point clouds under groundtruth alignment, with
       source keypoints and predited transformed coordinates, and a lines joining
       them (shorter lines means more accurate predictions)
    5: Source and target point clouds under predicted alignment (without clutter)

    Created 22 Oct 2021
    """

    if pose_gt is None:
        src_xyz_warped = src_xyz
        src_corr_warped = correspondences[:, :3]
    else:
        src_xyz_warped = se3_transform(pose_gt, src_xyz)
        src_corr_warped = se3_transform(pose_gt, correspondences[:, :3])

    vis = cvv.Visualizer(num_renderers=6, win_size=(1850, 1200))

    if correspondence_conf is None:
        src_kp_color = (255, 128, 128)
        tgt_kp_color = (128, 255, 128)
    else:
        conf = to_numpy(correspondence_conf)
        src_color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('autumn'))
        src_kp_color = (src_color_mapper.to_rgba(conf)[:, :3] * 255).astype(np.uint8)
        tgt_color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('summer'))
        tgt_kp_color = (tgt_color_mapper.to_rgba(conf)[:, :3] * 255).astype(np.uint8)
    # Show points on source
    vis.add_object(
        cvv.create_point_cloud(src_xyz_warped, colors=colors.RED),
        renderer_idx=0,
    )
    vis.add_object(
        cvv.create_point_cloud(src_corr_warped, colors=src_kp_color, pt_size=4),
        renderer_idx=0,
    )

    # Show points on target
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=3,
    )
    vis.add_object(
        cvv.create_point_cloud(correspondences[:, 3:], colors=tgt_kp_color, pt_size=4),
        renderer_idx=3,
    )

    # Show correspondences with lines joining the two
    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.RED),
        renderer_idx=1,
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=1,
    )
    vis.add_object(
        cvv.create_lines(correspondences),
        renderer_idx=1
    )

    # Show overlap using groundtruth pose
    vis.add_object(
        cvv.create_point_cloud(src_xyz_warped, colors=colors.RED),
        renderer_idx=4,
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=4,
    )
    vis.add_object(
        cvv.create_point_cloud(src_corr_warped, colors=src_kp_color, pt_size=4),
        renderer_idx=4
    )
    vis.add_object(
        cvv.create_point_cloud(correspondences[:, 3:], colors=tgt_kp_color, pt_size=4),
        renderer_idx=4
    )
    vis.add_object(
        cvv.create_lines(torch.cat([src_corr_warped, correspondences[:, 3:]], dim=1)),
        renderer_idx=4
    )

    # Show groundtruth (without clutter)
    vis.add_object(
        cvv.create_point_cloud(src_xyz_warped, colors=colors.RED),
        renderer_idx=2,
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=2,
    )

    # Show predicted pose
    if pose_pred is not None:
        vis.add_object(
            cvv.create_point_cloud(se3_transform(pose_pred, src_xyz), colors=colors.RED),
            renderer_idx=5,
        )
        vis.add_object(
            cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
            renderer_idx=5,
        )

    # Render loop
    vis.reset_camera()
    vis.start()
