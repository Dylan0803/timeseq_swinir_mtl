# 导入操作系统相关模块
import os
# 导入matplotlib绘图库
import matplotlib.pyplot as plt
# 导入时间模块
import time
# 导入数学函数模块
import math
# 导入HDF5文件读取模块
import h5py
# 导入数值计算库
import numpy as np
# 导入命令行参数解析模块
import argparse


# 这个函数保持不变，我们仍需要它来计算采样点坐标
def down_sample_2D_verbose(mat, scale_factor):
    """
    对2D矩阵进行下采样，生成低分辨率矩阵、稀疏矩阵和索引矩阵
    """
    scale_factor = math.floor(scale_factor)
    height, width = mat.shape
    output_height = height // scale_factor
    output_width = width // scale_factor
    res_height = height - output_height * scale_factor
    res_width = width - output_width * scale_factor
    start_heigh = res_height // 2
    start_width = res_width // 2
    output_lr_mat = np.zeros((output_height, output_width))
    output_lr_index_mat = np.zeros((output_height, output_width, 2))
    output_sparse_mat = np.zeros(mat.shape)
    for height_idx in range(output_height):
        for width_idx in range(output_width):
            hr_x_idx = start_heigh + height_idx * scale_factor
            hr_y_idx = start_width + width_idx * scale_factor
            output_lr_mat[height_idx, width_idx] = mat[hr_x_idx, hr_y_idx]
            output_sparse_mat[hr_x_idx, hr_y_idx] = mat[hr_x_idx, hr_y_idx]
            output_lr_index_mat[height_idx, width_idx, :] = np.array([hr_x_idx, hr_y_idx])
    return output_lr_mat, output_sparse_mat, output_lr_index_mat


#全新的数据加载函数，以匹配h5_dataset.py 逻辑
def load_data_from_swinir_h5(filename, wind_group, source_group, time_step, scale_factor):
    """
    根据 h5_dataset.py 的逻辑从H5文件中加载数据，并为KDM准备输入。

    :param filename: .h5 文件路径
    :param wind_group: 风场组名称 (e.g., 'wind1_0')
    :param source_group: 泄漏源组名称 (e.g., 's1')
    :param time_step: 时间步索引 (e.g., 1)
    :param scale_factor: 下采样比例，必须与生成LR数据时使用的一致
    :return: (gt_mat, lr_mat_from_file, sparse_mat_corrected, lr_index_mat)
    """
    print(f"Loading data: Wind={wind_group}, Source={source_group}, Time={time_step}")
    try:
        with h5py.File(filename, 'r') as f:
            # 检查路径是否存在
            if wind_group not in f:
                raise KeyError(f"Wind group '{wind_group}' not found in H5 file.")
            if source_group not in f[wind_group]:
                raise KeyError(f"Source group '{source_group}' not found in wind group '{wind_group}'.")

            # 1. 直接从文件读取 HR 和 LR 数据
            hr_dataset_name = f'HR_{time_step}'
            lr_dataset_name = f'LR_{time_step}'
            if hr_dataset_name not in f[wind_group][source_group]:
                 raise KeyError(f"Dataset '{hr_dataset_name}' not found.")
            if lr_dataset_name not in f[wind_group][source_group]:
                 raise KeyError(f"Dataset '{lr_dataset_name}' not found.")

            gt_mat = f[wind_group][source_group][hr_dataset_name][:]
            lr_mat_from_file = f[wind_group][source_group][lr_dataset_name][:]

            # 2. 调用 down_sample_2D_verbose，主要目的是为了获取采样坐标 `lr_index_mat`
            _, _, lr_index_mat = down_sample_2D_verbose(gt_mat, scale_factor)

            # 3. (关键步骤) 使用文件中的LR数据值 和 down_sample生成的坐标 来重建一个正确的稀疏矩阵
            sparse_mat_corrected = np.zeros_like(gt_mat)
            for h_idx in range(lr_index_mat.shape[0]):
                for w_idx in range(lr_index_mat.shape[1]):
                    # 获取在高分辨率图中的坐标
                    hr_coord = lr_index_mat[h_idx, w_idx, :].astype(int)
                    # 将文件中的低分辨率值，填充到稀疏矩阵的对应位置
                    sparse_mat_corrected[hr_coord[0], hr_coord[1]] = lr_mat_from_file[h_idx, w_idx]

            return gt_mat, lr_mat_from_file, sparse_mat_corrected, lr_index_mat

    except KeyError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None


# KDM核心算法函数保持不变
def get_gaussian_kdm_matrix(measure_mat,
                            mat_real_size,
                            sample_index_mat,
                            sample_conc_mat,
                            Rco, Gama):
    '''
    使用矩阵运算进行高斯核密度映射重建
    '''
    # 函数其余部分保持不变...
    assert len(measure_mat.shape) == 2, f'[get_gaussian_kdm_matrix] measure_mat should be a 2D matrix'
    assert len(sample_index_mat.shape) == 3, f'[get_gaussian_kdm_matrix] index_mat should be a 3D matrix'
    assert len(mat_real_size) == 2, f'[get_gaussian_kdm_matrix] mat_real_size should be a tuple with two elements'
    rows, columns = measure_mat.shape
    point_real_size = mat_real_size[0] / rows, mat_real_size[1] / columns
    samples_number = sample_index_mat.shape[0] * sample_index_mat.shape[1]
    x, y = np.mgrid[0:rows, 0:columns]
    mat_index = np.array(list(map(lambda xe, ye: [(ex, ey) for ex, ey in zip(xe, ye)], x, y)))
    sample_index_list = np.reshape(sample_index_mat, (-1, 2))
    sample_conc_list = np.reshape(sample_conc_mat, (-1,))
    sample_index_mat_extend = np.zeros((rows, columns, samples_number, 2))
    sample_index_mat_extend[:, :, :, 0] = np.tile(sample_index_list[:,0], (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    sample_index_mat_extend[:, :, :, 1] = np.tile(sample_index_list[:,1], (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    sample_conc_mat_extend = np.tile(sample_conc_list, (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    mat_index_extend = np.tile(mat_index, (samples_number, )).reshape(
        (rows, columns, 2, samples_number), order='F').transpose((0, 1, 3, 2))
    distance_matrix_index = mat_index_extend - sample_index_mat_extend
    distance_matrix_index[:, :, :, 0] = distance_matrix_index[:, :, :, 0] * point_real_size[0]
    distance_matrix_index[:, :, :, 1] = distance_matrix_index[:, :, :, 1] * point_real_size[1]
    distance_matrix = distance_matrix_index[:, :, :, 0] ** 2 + distance_matrix_index[:, :, :, 1] ** 2
    gama_sq = Gama ** 2
    if gama_sq == 0:
        gama_sq = 1e-12
    sample_conc_mat_extend = np.where(distance_matrix < Rco ** 2, sample_conc_mat_extend, 0)
    distance_matrix = np.where(distance_matrix < Rco ** 2, distance_matrix, 0)
    w_mat_extend = (1 / (2 * np. pi* gama_sq)) * np.exp(-(distance_matrix) / 2 / gama_sq)
    conc = w_mat_extend * sample_conc_mat_extend
    w_sum = w_mat_extend.sum(axis=2)
    conc_sum = conc.sum(axis=2)
    reconstruct_mat = np.divide(conc_sum, w_sum, out=np.zeros_like(conc_sum), where=w_sum!=0)
    return reconstruct_mat


# 可视化流程函数，更新了一下标题使其更精确
def gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat, rco_value=None):
    """
    可视化KDM的输入和输出（去除sparse_mat_corrected，只保留3个子图）
    """
    # 检查数据范围
    print(f"Visualization data ranges:")
    print(f"  GT: [{gt_mat.min():.6f}, {gt_mat.max():.6f}]")
    print(f"  LR: [{lr_mat.min():.6f}, {lr_mat.max():.6f}]")
    print(f"  Reconstruct: [{reconstruct_mat.min():.6f}, {reconstruct_mat.max():.6f}]")
    reconstruct_mat_clipped = np.clip(reconstruct_mat, gt_mat.min(), gt_mat.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 真实值
    im1 = axes[0].imshow(gt_mat, cmap='viridis')
    axes[0].set_title('Ground Truth (HR)')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 低分辨率图 (从文件加载)
    im2 = axes[1].imshow(lr_mat, cmap='viridis')
    axes[1].set_title('LR')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    # 第三个子图标题增加Rco值
    if rco_value is not None:
        axes[2].set_title(f'KDM (Rco={rco_value})')
    else:
        axes[2].set_title('KDM')
    im3 = axes[2].imshow(reconstruct_mat_clipped, cmap='viridis')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('gkdm_result.png', dpi=300, bbox_inches='tight')
    print("Image saved as gkdm_result.png")
    plt.show()


def calculate_normalized_mse(gt_mat, reconstruct_mat):
    """
    计算归一化MSE，将重建结果归一化到GT范围后再计算MSE
    """
    gt_range = gt_mat.max() - gt_mat.min()
    recon_range = reconstruct_mat.max() - reconstruct_mat.min()
    if recon_range > 0:
        reconstruct_normalized = (reconstruct_mat - reconstruct_mat.min()) / recon_range * gt_range + gt_mat.min()
    else:
        reconstruct_normalized = np.zeros_like(reconstruct_mat)
    mse_normalized = np.mean((gt_mat - reconstruct_normalized) ** 2)
    return mse_normalized


def test_rco_parameters(h5_file_path, wind_group, source_group, time_step, 
                       scale_factor, mat_real_size, rco_values):
    """
    测试不同Rco值对重建质量的影响，只返回归一化MSE
    """
    print("Starting Rco parameter test...")
    gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
        h5_file_path, wind_group, source_group, time_step, scale_factor
    )
    if gt_mat is None:
        print("Data loading failed")
        return [], []
    rco_list = []
    mse_normalized_list = []
    for rco in rco_values:
        gama = rco / 3.0
        reconstruct_mat = get_gaussian_kdm_matrix(
            measure_mat=sparse_mat,
            mat_real_size=mat_real_size,
            sample_index_mat=lr_index_mat,
            sample_conc_mat=lr_mat,
            Rco=rco,
            Gama=gama
        )
        mse_normalized = calculate_normalized_mse(gt_mat, reconstruct_mat)
        rco_list.append(rco)
        mse_normalized_list.append(mse_normalized)
        print(f"  Rco={rco}, Gama={gama:.4f}, MSE={mse_normalized:.6f}")
    return rco_list, mse_normalized_list


def plot_rco_vs_mse(rco_list, mse_normalized_list, save_path='rco_vs_mse.png'):
    """
    绘制Rco值与归一化MSE的关系图
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rco_list, mse_normalized_list, 'bo-', linewidth=2, markersize=8, label='MSE')
    best_mse_idx = np.argmin(mse_normalized_list)
    best_rco = rco_list[best_mse_idx]
    best_mse = mse_normalized_list[best_mse_idx]
    plt.plot(best_rco, best_mse, 'ro', markersize=12, 
             label=f'Best MSE (Rco={best_rco}, MSE={best_mse:.6f})')
    plt.xlabel('Rco Value', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Effect of Rco Value on MSE', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Rco vs MSE plot saved as {save_path}")
    plt.show()
    return best_rco, best_mse

# 主流程部分调用同步精简
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GKDM (Gaussian Kernel Density Mapping) Reconstruction Algorithm')
    parser.add_argument('--data_path', type=str, required=True,
                       help='H5 data file path (required)')
    args = parser.parse_args()
    h5_file_path = args.data_path
    wind_group_to_test = 'wind1_0'
    source_group_to_test = 's1'
    time_step_to_test = 50
    mat_real_size = (9.6, 9.6) 
    scale_factor = 6 
    # rco_values: 0到3，步长0.1
    rco_values = [round(x, 2) for x in np.arange(0, 3.01, 0.1)]
    print("--- Starting Rco Parameter Test ---")
    print(f"Data file path: {h5_file_path}")
    print(f"Using data: Wind={wind_group_to_test}, Source={source_group_to_test}, Time={time_step_to_test}")
    print(f"Testing Rco values: {rco_values}")
    rco_list, mse_normalized_list = test_rco_parameters(
        h5_file_path,
        wind_group_to_test,
        source_group_to_test,
        time_step_to_test,
        scale_factor,
        mat_real_size,
        rco_values
    )
    if rco_list:
        best_rco, best_mse = plot_rco_vs_mse(rco_list, mse_normalized_list)
        print(f"\n=== Test Results Summary ===")
        print(f"Best MSE Rco value: {best_rco}, MSE: {best_mse:.6f}")
        print(f"All results:")
        for i, rco in enumerate(rco_list):
            print(f"  Rco={rco}: MSE={mse_normalized_list[i]:.6f}")
        print(f"\nReconstructing with best parameters...")
        gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
            h5_file_path, wind_group_to_test, source_group_to_test, time_step_to_test, scale_factor
        )
        if gt_mat is not None:
            reconstruct_mat = get_gaussian_kdm_matrix(
                measure_mat=sparse_mat,
                mat_real_size=mat_real_size,
                sample_index_mat=lr_index_mat,
                sample_conc_mat=lr_mat,
                Rco=best_rco,
                Gama=best_rco/3.0
            )
            gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat, rco_value=best_rco)
    else:
        print("Parameter test failed")