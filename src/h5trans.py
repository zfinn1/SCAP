# import h5py
# import open3d as o3d
# import numpy as np
#
# # 读取H5文件
# with h5py.File('/mnt/disk/zjy/2_/PTT2-master/src/data/modelnet40_ply_hdf5_2048/ply_data_test1.h5', 'r') as f:
#     point_clouds = f['point_clouds'][:]  # (N_samples, N_points, 3)
#     labels = f['labels'][:]              # (N_samples,)
#
# # 提取第一个点云并保存为PLY
# sample_idx = 0
# point_cloud = point_clouds[sample_idx]
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud)
# o3d.io.write_point_cloud(f'data/modelnet_sample_{sample_idx}.ply', pcd)
#
# import h5py
# import numpy as np
# import open3d as o3d
#
#
# def h5_to_ply(h5_file_path, save_dir, sample_idx=0, is_train=True):
#     """
#     从ModelNet的H5文件中提取点云并保存为PLY格式
#     Args:
#         h5_file_path: H5文件路径（如'modelnet40_ply_hdf5_2048/ply_data_test0.h5'）
#         save_dir: PLY文件保存目录
#         sample_idx: 要提取的样本索引（H5文件中一个文件包含多个样本）
#         is_train: 是否为训练集（影响H5文件中的数据键名）
#     """
#     # 创建保存目录
#     import os
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 读取H5文件
#     with h5py.File(h5_file_path, 'r') as f:
#         # 训练集键名：'data'（点云）、'label'（标签）；测试集键名：'data_test'、'label_test'
#         data_key = 'data' if is_train else 'data_test'
#         label_key = 'label' if is_train else 'label_test'
#
#         # 提取点云坐标（shape: [N_samples, 2048, 3]）
#         point_clouds = f[data_key][:]  # (N, 2048, 3)
#         labels = f[label_key][:]  # (N, 1)
#
#         # 获取指定索引的点云
#         pc = point_clouds[sample_idx]  # (2048, 3)
#         label = labels[sample_idx][0]  # 类别标签（0-39）
#
#         # 保存为PLY文件
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pc)
#         save_path = os.path.join(save_dir, f"modelnet_test_{label}_{sample_idx}.ply")
#         o3d.io.write_point_cloud(save_path, pcd)
#         print(f"成功保存PLY文件：{save_path}")
#
#
# # 示例：提取测试集中第0个H5文件的第2个样本
# h5_to_ply(
#     h5_file_path="/mnt/disk/zjy/2_/PTT2-master/src/data/modelnet40_ply_hdf5_2048/ply_data_test0.h5",  # H5文件路径
#     save_dir="data/modelnet_demo_data",  # 保存目录（与Demo中路径一致）
#     sample_idx=2,  # 提取第2个样本
#     is_train=False  # 测试集
# )



import os
import h5py
import numpy as np
import open3d as o3d

def export_h5_to_ply(h5_file, out_dir, start_idx=0):
    """
    从 ModelNet h5 文件里导出点云并保存为 ply
    :param h5_file: h5 文件路径 (比如 test_0.h5)
    :param out_dir: 输出 ply 文件的目录
    :param start_idx: 保存文件的起始编号 (避免多个 h5 文件覆盖)
    """
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_file, 'r') as f:
        data = f['data'][:]  # [N, 2048, 3]
        labels = f['label'][:]  # [N, 1]

    for i in range(data.shape[0]):
        points = data[i]  # [2048, 3]
        label = labels[i][0]

        # 保存成 ply
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        filename = os.path.join(out_dir, f'modelnet_test_{start_idx+i}_{label}.ply')
        o3d.io.write_point_cloud(filename, pcd)

    print(f"✅ Exported {data.shape[0]} point clouds from {h5_file} to {out_dir}")
    return start_idx + data.shape[0]


if __name__ == "__main__":
    # 示例：把 test split 里的所有点云导出来
    h5_dir = "/mnt/disk/zjy/2_/PTT2-master/src/data/modelnet40_ply_hdf5_2048/"
    out_dir = "data/modelnet_demo_data/"



    # test 文件列表（和 dataset 里对应）
    test_files = [os.path.join(h5_dir, f"ply_data_test{i}.h5") for i in range(5)]  # test_0.h5 ~ test_4.h5
    idx = 0
    for f in test_files:
        idx = export_h5_to_ply(f, out_dir, start_idx=idx)
