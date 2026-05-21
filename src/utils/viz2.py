# utils/viz.py （只替换这两个函数）
import numpy as np
import torch
import scipy.io as sio
import os
import matplotlib.pyplot as plt

def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)

def apply_se3(T, pts, pose_index=None):
    """
    Robust apply SE3 transform.
    T may be:
      - None -> return pts
      - (4,4) single transform
      - (B,4,4) batched transforms -> use pose_index if provided, else use first with a warning
    pts: (N,3) numpy array
    """
    if T is None:
        return pts
    Tn = to_numpy(T)
    if Tn.ndim == 2 and Tn.shape == (4,4):
        T_use = Tn
    elif Tn.ndim == 3 and Tn.shape[1:] == (4,4):
        # batched transforms
        if pose_index is not None:
            T_use = Tn[int(pose_index)]
        else:
            # fallback: take first and warn (this is better than crashing)
            # If you expect a matching index, pass pose_index from the caller.
            print("[apply_se3] Warning: received batched pose (B,4,4) but no pose_index given. Using T[0].")
            T_use = Tn[0]
    else:
        # support common alternative: (4,) flattened or (16,) vector? reject clearly
        raise ValueError(f"pose must be 4x4 or (B,4,4). got shape {Tn.shape}")
    R = T_use[:3, :3]
    t = T_use[:3, 3]
    pts = to_numpy(pts)
    return (R @ pts.T).T + t

# 以下为之前给你的 save_registration_mat函数的改良版（增加 pose_index 参数）
def save_registration_mat(src_xyz, tgt_xyz, correspondences,
                          correspondence_conf=None,
                          pose_gt=None, pose_pred=None,
                          out_mat='./vis6.mat',
                          subsample_pts=3000,
                          max_lines=500,
                          pose_index=None):
    """
    保存用于 Matlab 恢复 6 个视图的 .mat 文件（更鲁棒：pose_gt/pose_pred 支持 batched）
    参数 pose_index: 当 pose_gt/pose_pred 是 batched (B,4,4) 时，指定使用哪一个（int）。
    如果 pose_index 为 None 且 pose 是 batched，会默认使用 [0] 并打印警告。
    """
    def _maybe_subsample(pts, max_pts=3000, seed=0):
        if pts is None or pts.size == 0:
            return np.zeros((0,3), dtype=np.float32)
        n = pts.shape[0]
        if n <= max_pts:
            return pts.copy()
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=max_pts, replace=False)
        return pts[idx]

    def _colormap_rgb(conf, cmap_name='autumn'):
        cmap = plt.get_cmap(cmap_name)
        conf = np.array(conf)
        if conf.size == 0:
            return np.zeros((0,3), dtype=np.float32)
        if conf.max() - conf.min() < 1e-9:
            conf_norm = np.clip(conf, 0.0, 1.0)
        else:
            conf_norm = (conf - conf.min()) / (conf.max() - conf.min())
        rgb = cmap(conf_norm)[:, :3]
        return rgb.astype(np.float32)

    src = to_numpy(src_xyz).astype(np.float32)
    tgt = to_numpy(tgt_xyz).astype(np.float32)
    corr = to_numpy(correspondences).astype(np.float32) if correspondences is not None else np.zeros((0,6), dtype=np.float32)
    conf = to_numpy(correspondence_conf).astype(np.float32) if correspondence_conf is not None else np.zeros((corr.shape[0],), dtype=np.float32)

    # compute transforms robustly (select item if batched)
    src_gt = apply_se3(pose_gt, src, pose_index=pose_index) if pose_gt is not None else None
    src_pred = apply_se3(pose_pred, src, pose_index=pose_index) if pose_pred is not None else None

    src_plot = _maybe_subsample(src, max_pts=subsample_pts, seed=1)
    tgt_plot = _maybe_subsample(tgt, max_pts=subsample_pts, seed=2)
    src_gt_plot = _maybe_subsample(src_gt, max_pts=subsample_pts, seed=3) if src_gt is not None else np.zeros((0,3), dtype=np.float32)
    src_pred_plot = _maybe_subsample(src_pred, max_pts=subsample_pts, seed=5) if src_pred is not None else np.zeros((0,3), dtype=np.float32)

    K = corr.shape[0]
    if K == 0:
        line_idx = np.zeros((0,), dtype=np.int32)
    else:
        line_idx = np.arange(K) if K <= max_lines else np.random.choice(K, size=max_lines, replace=False)
    corr_lines = corr[line_idx] if K>0 else np.zeros((0,6), dtype=np.float32)
    conf_lines = conf[line_idx] if (conf is not None and conf.size>0) else np.zeros((corr_lines.shape[0],), dtype=np.float32)

    src_kp_colors = _colormap_rgb(conf, 'autumn') if conf.size>0 else np.zeros((corr.shape[0],3), dtype=np.float32)
    tgt_kp_colors = _colormap_rgb(conf, 'summer') if conf.size>0 else np.zeros((corr.shape[0],3), dtype=np.float32)
    line_colors = _colormap_rgb(conf_lines, 'gray') if conf_lines.size>0 else np.tile(np.array([[0.5,0.5,0.5]], dtype=np.float32), (corr_lines.shape[0],1))

    all_pts = np.concatenate([src.reshape(-1,3), tgt.reshape(-1,3)], axis=0)
    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)
    axis_lims = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]], dtype=np.float32)

    matdict = {}
    matdict['src_full'] = src
    matdict['tgt_full'] = tgt
    matdict['corr_full'] = corr
    matdict['corr_conf'] = conf
    matdict['src_gt_full'] = src_gt if src_gt is not None else np.zeros((0,3), dtype=np.float32)
    matdict['src_pred_full'] = src_pred if src_pred is not None else np.zeros((0,3), dtype=np.float32)
    matdict['axis_lims'] = axis_lims

    matdict['p0_src_plot'] = src_plot
    matdict['p0_src_kps'] = corr[:,:3]

    matdict['p1_src_plot'] = src_plot
    matdict['p1_tgt_plot'] = tgt_plot
    matdict['p1_lines'] = corr_lines
    matdict['p1_line_idx'] = line_idx
    matdict['p1_line_colors'] = line_colors

    matdict['p2_src_gt_plot'] = src_gt_plot
    matdict['p2_tgt_plot'] = tgt_plot

    matdict['p3_tgt_plot'] = tgt_plot
    matdict['p3_tgt_kps'] = corr[:, 3:]

    matdict['p4_src_gt_plot'] = src_gt_plot
    matdict['p4_tgt_plot'] = tgt_plot
    matdict['p4_src_kps'] = corr[:,:3]
    matdict['p4_tgt_kps'] = corr[:,3:]
    matdict['p4_lines'] = corr_lines
    matdict['p4_line_colors'] = line_colors

    matdict['p5_src_pred_plot'] = src_pred_plot
    matdict['p5_tgt_plot'] = tgt_plot

    matdict['src_kp_colors_full'] = src_kp_colors
    matdict['tgt_kp_colors_full'] = tgt_kp_colors
    matdict['p1_conf_lines'] = conf_lines
    matdict['corr_lines_idx'] = line_idx

    os.makedirs(os.path.dirname(out_mat) or '.', exist_ok=True)
    sio.savemat(out_mat, matdict, do_compression=True)
    print(f"[save_registration_mat] saved to {out_mat} (pose_index={pose_index})")
    return out_mat
