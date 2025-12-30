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
            output_lr_index_mat[height_idx, width_idx,
                                :] = np.array([hr_x_idx, hr_y_idx])
    return output_lr_mat, output_sparse_mat, output_lr_index_mat


# 全新的数据加载函数，以匹配h5_dataset.py 逻辑
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
    try:
        with h5py.File(filename, 'r') as f:
            # 检查路径是否存在
            if wind_group not in f:
                raise KeyError(
                    f"Wind group '{wind_group}' not found in H5 file.")
            if source_group not in f[wind_group]:
                raise KeyError(
                    f"Source group '{source_group}' not found in wind group '{wind_group}'.")

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
                    sparse_mat_corrected[hr_coord[0], hr_coord[1]
                                         ] = lr_mat_from_file[h_idx, w_idx]

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
    assert len(
        measure_mat.shape) == 2, f'[get_gaussian_kdm_matrix] measure_mat should be a 2D matrix'
    assert len(
        sample_index_mat.shape) == 3, f'[get_gaussian_kdm_matrix] index_mat should be a 3D matrix'
    assert len(
        mat_real_size) == 2, f'[get_gaussian_kdm_matrix] mat_real_size should be a tuple with two elements'
    rows, columns = measure_mat.shape
    point_real_size = mat_real_size[0] / rows, mat_real_size[1] / columns
    samples_number = sample_index_mat.shape[0] * sample_index_mat.shape[1]
    x, y = np.mgrid[0:rows, 0:columns]
    mat_index = np.array(
        list(map(lambda xe, ye: [(ex, ey) for ex, ey in zip(xe, ye)], x, y)))
    sample_index_list = np.reshape(sample_index_mat, (-1, 2))
    sample_conc_list = np.reshape(sample_conc_mat, (-1,))
    sample_index_mat_extend = np.zeros((rows, columns, samples_number, 2))
    sample_index_mat_extend[:, :, :, 0] = np.tile(sample_index_list[:, 0], (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    sample_index_mat_extend[:, :, :, 1] = np.tile(sample_index_list[:, 1], (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    sample_conc_mat_extend = np.tile(sample_conc_list, (rows, columns)).reshape(
        (rows, columns, samples_number), order='C')
    mat_index_extend = np.tile(mat_index, (samples_number, )).reshape(
        (rows, columns, 2, samples_number), order='F').transpose((0, 1, 3, 2))
    distance_matrix_index = mat_index_extend - sample_index_mat_extend
    distance_matrix_index[:, :, :, 0] = distance_matrix_index[:,
                                                              :, :, 0] * point_real_size[0]
    distance_matrix_index[:, :, :, 1] = distance_matrix_index[:,
                                                              :, :, 1] * point_real_size[1]
    distance_matrix = distance_matrix_index[:, :, :,
                                            0] ** 2 + distance_matrix_index[:, :, :, 1] ** 2
    gama_sq = Gama ** 2
    if gama_sq == 0:
        gama_sq = 1e-12
    sample_conc_mat_extend = np.where(
        distance_matrix < Rco ** 2, sample_conc_mat_extend, 0)
    distance_matrix = np.where(distance_matrix < Rco ** 2, distance_matrix, 0)
    w_mat_extend = (1 / (2 * np. pi * gama_sq)) * \
        np.exp(-(distance_matrix) / 2 / gama_sq)
    conc = w_mat_extend * sample_conc_mat_extend
    w_sum = w_mat_extend.sum(axis=2)
    conc_sum = conc.sum(axis=2)
    reconstruct_mat = np.divide(
        conc_sum, w_sum, out=np.zeros_like(conc_sum), where=w_sum != 0)
    return reconstruct_mat


def gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat, rco_value=None, save_path=None):
    """
    可视化KDM的输入和输出（去除sparse_mat_corrected，只保留3个子图）
    不在每个子图右侧显示图例（颜色条）
    """
    if save_path is None:
        # 默认保存到与脚本同路径的 results/gkdm_results 目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, 'results', 'gkdm_results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'gkdm_result.png')

    # 检查数据范围
    print(f"Visualization data ranges:")
    print(f"  GT: [{gt_mat.min():.6f}, {gt_mat.max():.6f}]")
    print(f"  LR: [{lr_mat.min():.6f}, {lr_mat.max():.6f}]")
    print(
        f"  Reconstruct: [{reconstruct_mat.min():.6f}, {reconstruct_mat.max():.6f}]")
    reconstruct_mat_clipped = np.clip(
        reconstruct_mat, gt_mat.min(), gt_mat.max())

    # 创建组合图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 真实值
    im1 = axes[0].imshow(gt_mat, cmap='viridis')
    axes[0].text(0.5, -0.08, '(a) Ground Truth (HR)', transform=axes[0].transAxes,
                 ha='center', va='top', fontsize=14)

    # 低分辨率图 (从文件加载)
    im2 = axes[1].imshow(lr_mat, cmap='viridis')
    axes[1].text(0.5, -0.08, '(b) LR', transform=axes[1].transAxes,
                 ha='center', va='top', fontsize=14)

    # 第三个子图标题增加Rco值
    if rco_value is not None:
        axes[2].text(0.5, -0.08, f'(c) KDM (Rco={rco_value})', transform=axes[2].transAxes,
                     ha='center', va='top', fontsize=14)
    else:
        axes[2].text(0.5, -0.08, '(c) KDM', transform=axes[2].transAxes,
                     ha='center', va='top', fontsize=14)
    im3 = axes[2].imshow(reconstruct_mat_clipped, cmap='viridis')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    # 确保底部题注不被裁剪
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved as {save_path}")

    # 单独保存三个子图（无图注）
    save_dir = os.path.dirname(
        save_path) if os.path.dirname(save_path) else '.'

    # 保存 Ground Truth
    plt.figure(figsize=(5, 5))
    plt.imshow(gt_mat, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    gt_save_path = os.path.join(save_dir, 'gkdm_gt.png')
    plt.savefig(gt_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ground Truth saved as {gt_save_path}")

    # 保存 LR
    plt.figure(figsize=(5, 5))
    plt.imshow(lr_mat, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    lr_save_path = os.path.join(save_dir, 'gkdm_lr.png')
    plt.savefig(lr_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"LR saved as {lr_save_path}")

    # 保存 KDM 重建结果
    plt.figure(figsize=(5, 5))
    plt.imshow(reconstruct_mat_clipped, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    kdm_save_path = os.path.join(save_dir, 'gkdm_reconstruct.png')
    plt.savefig(kdm_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"KDM Reconstruct saved as {kdm_save_path}")

    # 保存用于后期处理的数据（CSV），便于在 Origin 等软件中进一步调整
    try:
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        gt_csv_path = os.path.join(save_dir, f"{base_name}_GT.csv")
        lr_csv_path = os.path.join(save_dir, f"{base_name}_LR.csv")
        recon_csv_path = os.path.join(save_dir, f"{base_name}_Reconstruct.csv")
        np.savetxt(gt_csv_path, gt_mat, delimiter=',', fmt='%.10g')
        np.savetxt(lr_csv_path, lr_mat, delimiter=',', fmt='%.10g')
        np.savetxt(recon_csv_path, reconstruct_mat_clipped,
                   delimiter=',', fmt='%.10g')
        print(f"Data saved: {gt_csv_path}, {lr_csv_path}, {recon_csv_path}")
    except Exception as e:
        print(f"Warning: failed to save CSV data for gkdm_result: {e}")

    plt.show()


def calculate_normalized_mse(gt_mat, reconstruct_mat):
    """
    计算归一化MSE，将重建结果归一化到GT范围后再计算MSE
    """
    gt_range = gt_mat.max() - gt_mat.min()
    recon_range = reconstruct_mat.max() - reconstruct_mat.min()
    if recon_range > 0:
        reconstruct_normalized = (
            reconstruct_mat - reconstruct_mat.min()) / recon_range * gt_range + gt_mat.min()
    else:
        reconstruct_normalized = np.zeros_like(reconstruct_mat)
    mse_normalized = np.mean((gt_mat - reconstruct_normalized) ** 2)
    return mse_normalized


def mse_to_psnr(gt_mat, mse_normalized):
    """
    基于已对齐到 GT 动态范围后的 MSE 计算 PSNR
    PSNR = 10 * log10( (peak^2) / MSE ), 其中 peak = max(GT) - min(GT)
    """
    peak = gt_mat.max() - gt_mat.min()
    if peak <= 0:
        return float('inf') if mse_normalized == 0 else 0.0
    if mse_normalized <= 0:
        return float('inf')
    return 10.0 * np.log10((peak * peak) / mse_normalized)


def test_kernel_width_parameters(h5_file_path, wind_group, source_group, time_step,
                                 scale_factor, mat_real_size, kernel_width_values):
    """
    测试不同核宽值对重建质量的影响，返回核宽与PSNR的关系

    注意：核宽参数对应原始代码中的σ(Gama)参数
    - 核宽 = σ = Gama (高斯核的宽度)
    - Rco = 核宽 * 3 (截止半径，影响重建范围)
    """
    print("Starting kernel width parameter test...")
    print("Note: Kernel width corresponds to σ(Gama) in original code, with Rco = kernel_width * 3")
    gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
        h5_file_path, wind_group, source_group, time_step, scale_factor
    )
    if gt_mat is None:
        print("Data loading failed")
        return [], []

    kernel_width_list = []
    psnr_list = []
    for kernel_width in kernel_width_values:
        # 核宽 = σ = Gama，Rco = 核宽 * 3
        rco = kernel_width * 3
        reconstruct_mat = get_gaussian_kdm_matrix(
            measure_mat=sparse_mat,
            mat_real_size=mat_real_size,
            sample_index_mat=lr_index_mat,
            sample_conc_mat=lr_mat,
            Rco=rco,
            Gama=kernel_width
        )
        mse_normalized = calculate_normalized_mse(gt_mat, reconstruct_mat)
        psnr = mse_to_psnr(gt_mat, mse_normalized)
        kernel_width_list.append(kernel_width)
        psnr_list.append(psnr)
        print(
            f"  Kernel Width(σ)={kernel_width}, Rco={rco:.4f}, PSNR={psnr:.4f} dB")
    return kernel_width_list, psnr_list


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GKDM (Gaussian Kernel Density Mapping) Reconstruction Algorithm')

    # 必需参数
    parser.add_argument('--data_path', type=str, required=True,
                        help='H5 data file path (required)')

    # 保存目录参数
    parser.add_argument('--save_dir', type=str, default='gkdm_results',
                        help='Results save directory')

    # 测试模式选择
    parser.add_argument('--test_mode', type=str, default='generalization',
                        choices=['generalization', 'test_set',
                                 'all_generalization', 'all_test_set'],
                        help='Test mode: generalization, test_set, all_generalization, all_test_set')

    # 样本选择参数
    parser.add_argument('--sample_specs', type=str, default=None,
                        help='Generalization test sample specs, separated by semicolon, e.g.: wind1_0,s1,50;wind2_0,s2,30')
    parser.add_argument('--test_indices', type=str, default=None,
                        help='Test set indices, separated by comma, e.g.: 1,2,3,4,5')

    # 核宽参数范围（对应原始代码中的σ(Gama)参数，DEFAULT_GAMA=DEFAULT_RCO/3≈0.67）
    parser.add_argument('--kernel_width_start', type=float, default=0.2,
                        help='Start value for kernel width range (corresponds to σ(Gama) in original code)')
    parser.add_argument('--kernel_width_end', type=float, default=1.5,
                        help='End value for kernel width range (corresponds to σ(Gama) in original code)')
    parser.add_argument('--kernel_width_step', type=float, default=0.1,
                        help='Step size for kernel width range (corresponds to σ(Gama) in original code)')

    # 物理参数
    parser.add_argument('--mat_width', type=float, default=9.6,
                        help='Physical width (meters)')
    parser.add_argument('--mat_height', type=float, default=9.6,
                        help='Physical height (meters)')
    parser.add_argument('--scale_factor', type=int, default=6,
                        help='Downsampling scale factor')

    args = parser.parse_args()
    return args


def get_sample_indices_from_specs(sample_specs_str, h5_file_path):
    """从样本规格字符串中获取样本索引"""
    if not sample_specs_str:
        return []

    sample_specs = [spec.strip() for spec in sample_specs_str.split(';')]
    indices = []

    try:
        with h5py.File(h5_file_path, 'r') as f:
            for spec in sample_specs:
                wind_group, source_group, time_step = spec.split(',')
                wind_group = wind_group.strip()
                source_group = source_group.strip()
                time_step = int(time_step.strip())

                # 检查数据是否存在
                if wind_group in f and source_group in f[wind_group]:
                    hr_key = f'HR_{time_step}'
                    lr_key = f'LR_{time_step}'
                    if hr_key in f[wind_group][source_group] and lr_key in f[wind_group][source_group]:
                        indices.append({
                            'wind_group': wind_group,
                            'source_group': source_group,
                            'time_step': time_step
                        })
                        print(
                            f"Found sample: {wind_group}, {source_group}, {time_step}")
                    else:
                        print(
                            f"Warning: Data not found for {wind_group}, {source_group}, {time_step}")
                else:
                    print(
                        f"Warning: Group not found for {wind_group}, {source_group}")

    except Exception as e:
        print(f"Error reading H5 file: {e}")
        return []

    return indices


def get_test_set_indices(test_indices_str, h5_file_path):
    """从测试集索引字符串中获取样本索引"""
    if not test_indices_str:
        return []

    try:
        # 解析测试集索引
        indices = [int(idx.strip()) for idx in test_indices_str.split(',')]

        # 加载数据集获取测试集
        from datasets.h5_dataset import generate_train_valid_test_dataset
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            h5_file_path, seed=42)

        # 获取测试集的样本信息
        sample_indices = []
        for idx in indices:
            if idx < len(test_dataset.data_indices):
                sample_info = test_dataset.data_indices[idx]
                sample_indices.append(sample_info)
                print(f"Found test set sample {idx}: {sample_info}")
            else:
                print(f"Warning: Test index {idx} out of range")

        return sample_indices

    except Exception as e:
        print(f"Error processing test set indices: {e}")
        return []


def evaluate_multiple_samples(h5_file_path, sample_indices, kernel_width_values, mat_real_size, scale_factor, save_dir):
    """评估多个样本，计算每个核宽的平均PSNR，选择平均PSNR最大的核宽"""
    print(f"Evaluating {len(sample_indices)} samples...")

    all_psnr_results = {}
    for kernel_width in kernel_width_values:
        all_psnr_results[kernel_width] = []

    # 记录每个样本的最佳核宽值（用于对比分析）
    sample_best_kernel_widths = []

    for i, sample_info in enumerate(sample_indices):
        print(f"\n--- Processing Sample {i+1}/{len(sample_indices)} ---")
        print(f"Sample info: {sample_info}")

        # 加载数据
        gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
            h5_file_path,
            sample_info['wind_group'],
            sample_info['source_group'],
            sample_info['time_step'],
            scale_factor
        )

        if gt_mat is None:
            print(f"Failed to load data for sample {i+1}")
            continue

        # 测试不同核宽值，记录这个样本的所有PSNR
        sample_psnr_results = {}
        print(f"Testing kernel width values for sample {i+1}:")
        for kernel_width in kernel_width_values:
            rco = kernel_width * 3
            reconstruct_mat = get_gaussian_kdm_matrix(
                measure_mat=sparse_mat,
                mat_real_size=mat_real_size,
                sample_index_mat=lr_index_mat,
                sample_conc_mat=lr_mat,
                Rco=rco,
                Gama=kernel_width
            )

            mse_normalized = calculate_normalized_mse(gt_mat, reconstruct_mat)
            psnr = mse_to_psnr(gt_mat, mse_normalized)

            all_psnr_results[kernel_width].append(psnr)
            sample_psnr_results[kernel_width] = psnr
            print(
                f"  Kernel Width(σ)={kernel_width:4.2f}, Rco={rco:.4f}, PSNR={psnr:.4f} dB")

        # 找到这个样本的最佳核宽值（用于对比分析）
        best_kernel_width_for_sample = max(
            sample_psnr_results, key=sample_psnr_results.get)
        best_psnr_for_sample = sample_psnr_results[best_kernel_width_for_sample]
        sample_best_kernel_widths.append(best_kernel_width_for_sample)
        print(
            f"  => Best kernel width for this sample: {best_kernel_width_for_sample} (PSNR={best_psnr_for_sample:.4f} dB)")

    # 计算每个核宽的平均PSNR
    avg_psnr_per_kernel_width = {}
    print("\n--- Calculating Average PSNR for Each Kernel Width ---")
    for kernel_width in kernel_width_values:
        if all_psnr_results[kernel_width]:  # 确保有数据
            avg_psnr_per_kernel_width[kernel_width] = np.mean(
                all_psnr_results[kernel_width])
            print(
                f"Kernel Width={kernel_width:4.2f}: Average PSNR = {avg_psnr_per_kernel_width[kernel_width]:.4f} dB (based on {len(all_psnr_results[kernel_width])} samples)")
        else:
            avg_psnr_per_kernel_width[kernel_width] = float('-inf')
            print(f"Kernel Width={kernel_width:4.2f}: No data available")

    # 找到平均PSNR最大的核宽值
    best_kernel_width = max(avg_psnr_per_kernel_width,
                            key=avg_psnr_per_kernel_width.get)
    best_avg_psnr = avg_psnr_per_kernel_width[best_kernel_width]

    # 统计每个核宽值被选为最佳的次数（用于对比分析）
    kernel_width_counts = {}
    for kernel_width in kernel_width_values:
        kernel_width_counts[kernel_width] = sample_best_kernel_widths.count(
            kernel_width)

    print(f"\n=== Final Selection ===")
    print(
        f"Best kernel width by average PSNR: {best_kernel_width} (Average PSNR={best_avg_psnr:.4f} dB)")

    return best_kernel_width, best_avg_psnr, kernel_width_counts, sample_best_kernel_widths, avg_psnr_per_kernel_width


def plot_kernel_width_vs_psnr(kernel_width_list, psnr_list, save_path=None):
    """绘制核宽值与PSNR的关系图"""
    if save_path is None:
        # 默认保存到与脚本同路径的 results/gkdm_results 目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, 'results', 'gkdm_results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'kernel_width_vs_psnr.png')

    plt.figure(figsize=(10, 6))
    plt.plot(kernel_width_list, psnr_list, 'bo-',
             linewidth=2, markersize=8, label='PSNR')
    best_psnr_idx = np.argmax(psnr_list)
    best_kernel_width = kernel_width_list[best_psnr_idx]
    best_psnr = psnr_list[best_psnr_idx]
    plt.plot(best_kernel_width, best_psnr, 'ro', markersize=12,
             label=f'Best PSNR (Kernel Width={best_kernel_width}, PSNR={best_psnr:.4f} dB)')
    plt.xlabel('Kernel Width', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Effect of Kernel Width on PSNR', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Kernel Width vs PSNR plot saved as {save_path}")

    # 保存曲线数据为 CSV，便于在 Origin 等软件中重绘
    try:
        save_dir = os.path.dirname(
            save_path) if os.path.dirname(save_path) else '.'
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        csv_path = os.path.join(save_dir, f"{base_name}_data.csv")
        data_mat = np.column_stack(
            (np.asarray(kernel_width_list), np.asarray(psnr_list)))
        np.savetxt(csv_path, data_mat, delimiter=',',
                   fmt='%.10g', header='Kernel_Width,PSNR', comments='')
        print(f"Data saved: {csv_path}")
    except Exception as e:
        print(
            f"Warning: failed to save CSV data for kernel_width_vs_psnr: {e}")

    plt.show()
    return best_kernel_width, best_psnr


def plot_average_psnr_vs_kernel_width(kernel_width_values, avg_psnr_per_kernel_width, best_kernel_width, save_path=None):
    """绘制平均PSNR vs 核宽的关系图"""
    if save_path is None:
        # 默认保存到与脚本同路径的 results/gkdm_results 目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, 'results', 'gkdm_results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(
            results_dir, 'average_psnr_vs_kernel_width.png')

    plt.figure(figsize=(10, 6))

    # 准备数据
    kernel_width_list = sorted(kernel_width_values)
    psnr_list = [avg_psnr_per_kernel_width[kw] for kw in kernel_width_list]

    plt.plot(kernel_width_list, psnr_list, 'bo-', linewidth=2,
             markersize=8, label='Average PSNR')

    # 标记最佳核宽值
    best_psnr = avg_psnr_per_kernel_width[best_kernel_width]
    plt.plot(best_kernel_width, best_psnr, 'ro', markersize=12,
             label=f'Best Average PSNR (Kernel Width={best_kernel_width}, PSNR={best_psnr:.4f} dB)')

    plt.xlabel('Kernel Width', fontsize=12)
    plt.ylabel('Average PSNR (dB)', fontsize=12)
    plt.title('Average PSNR vs Kernel Width Across All Samples', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Average PSNR vs Kernel Width plot saved as {save_path}")
    plt.show()

    return best_kernel_width, best_psnr


def reconstruct_and_save_for_samples(h5_file_path, sample_indices, scale_factor, mat_real_size, kernel_width, out_root):
    os.makedirs(out_root, exist_ok=True)
    for i, sample_info in enumerate(sample_indices):
        gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
            h5_file_path,
            sample_info['wind_group'],
            sample_info['source_group'],
            sample_info['time_step'],
            scale_factor
        )
        if gt_mat is None:
            print(f"[Skip] Failed to load sample {i}: {sample_info}")
            continue

        # 每个样本单独的子目录（含更可读的命名）
        slug = f"{sample_info['wind_group']}_{sample_info['source_group']}_{sample_info['time_step']}"
        sample_dir = os.path.join(out_root, f"sample_{i}_{slug}")
        os.makedirs(sample_dir, exist_ok=True)

        # KDM 重建
        rco = kernel_width * 3
        reconstruct_mat = get_gaussian_kdm_matrix(
            measure_mat=sparse_mat,
            mat_real_size=mat_real_size,
            sample_index_mat=lr_index_mat,
            sample_conc_mat=lr_mat,
            Rco=rco,
            Gama=kernel_width
        )

        # 组合图保存到该样本目录；gkdm_flow 内部会基于 save_path 基名生成对应 CSV
        save_path = os.path.join(sample_dir, "gkdm_result.png")
        gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat,
                  rco_value=rco, save_path=save_path)

        print(f"[OK] Saved: {sample_dir}")


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gkdm_results_dir = os.path.join(script_dir, 'results', 'gkdm_results')
    os.makedirs(gkdm_results_dir, exist_ok=True)

    h5_file_path = args.data_path
    mat_real_size = (args.mat_width, args.mat_height)
    scale_factor = args.scale_factor

    # 生成核宽列表
    kernel_width_values = [round(x, 2) for x in np.arange(
        args.kernel_width_start, args.kernel_width_end + args.kernel_width_step, args.kernel_width_step)]

    # 按 test_mode 获取 sample_indices（保持你原有的逻辑）
    sample_indices = []
    if args.test_mode == 'generalization':
        if args.sample_specs:
            sample_indices = get_sample_indices_from_specs(
                args.sample_specs, h5_file_path)
            print(
                f"Using generalization test mode with {len(sample_indices)} samples")
        else:
            print("Error: generalization test mode requires sample_specs parameter")
            exit(1)
    elif args.test_mode == 'test_set':
        if args.test_indices:
            sample_indices = get_test_set_indices(
                args.test_indices, h5_file_path)
            print(f"Using test set mode with {len(sample_indices)} samples")
        else:
            print("Error: test_set mode requires test_indices parameter")
            exit(1)
    elif args.test_mode == 'all_generalization':
        # 获取所有可用的样本
        sample_indices = get_sample_indices_from_specs(
            "wind1_0,s1,50;wind1_0,s2,50;wind2_0,s1,50;wind2_0,s2,50;wind3_0,s1,50;wind3_0,s2,50", h5_file_path)
        print(
            f"Using all generalization test mode with {len(sample_indices)} samples")
    elif args.test_mode == 'all_test_set':
        # 获取所有测试集样本
        from datasets.h5_dataset import generate_train_valid_test_dataset
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            h5_file_path, seed=42)
        sample_indices = test_dataset.data_indices
        print(f"Using all test set mode with {len(sample_indices)} samples")
    else:
        print("Using single sample test mode")
        # 使用默认样本
        sample_indices = [{
            'wind_group': 'wind1_0',
            'source_group': 's1',
            'time_step': 50
        }]

    if not sample_indices:
        print("No samples to evaluate!")
        exit(1)

    # 分支1：固定核宽且多样本 → 逐个样本输出
    if len(kernel_width_values) == 1 and len(sample_indices) > 1:
        reconstruct_and_save_for_samples(
            h5_file_path=h5_file_path,
            sample_indices=sample_indices,
            scale_factor=scale_factor,
            mat_real_size=mat_real_size,
            kernel_width=kernel_width_values[0],
            out_root=gkdm_results_dir
        )
        print(
            f"Done: saved {len(sample_indices)} samples to {gkdm_results_dir}")
    else:
        # 分支2：保持你原有的单样本/多样本扫描与平均流程
        if len(sample_indices) > 1:
            best_kernel_width, avg_psnr, kernel_width_counts, sample_best_kernel_widths, avg_psnr_per_kernel_width = evaluate_multiple_samples(
                h5_file_path, sample_indices, kernel_width_values, mat_real_size, scale_factor, args.save_dir)
            plot_average_psnr_vs_kernel_width(kernel_width_values, avg_psnr_per_kernel_width, best_kernel_width, os.path.join(
                gkdm_results_dir, 'average_psnr_vs_kernel_width.png'))
            # 用 best_kernel_width 做一次可视化（可选）
            sample_info = sample_indices[0]
            gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
                h5_file_path, sample_info['wind_group'], sample_info['source_group'], sample_info['time_step'], scale_factor
            )
            if gt_mat is not None:
                rco = best_kernel_width * 3
                reconstruct_mat = get_gaussian_kdm_matrix(
                    measure_mat=sparse_mat,
                    mat_real_size=mat_real_size,
                    sample_index_mat=lr_index_mat,
                    sample_conc_mat=lr_mat,
                    Rco=rco,
                    Gama=best_kernel_width
                )
                gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat, rco_value=rco,
                          save_path=os.path.join(gkdm_results_dir, 'gkdm_result.png'))
        else:
            # 单样本参数扫描（保持原逻辑）
            sample_info = sample_indices[0]
            kernel_width_list, psnr_list = test_kernel_width_parameters(
                h5_file_path, sample_info['wind_group'], sample_info['source_group'], sample_info['time_step'],
                scale_factor, mat_real_size, kernel_width_values
            )
            if kernel_width_list:
                best_kernel_width, best_psnr = plot_kernel_width_vs_psnr(
                    kernel_width_list, psnr_list, os.path.join(gkdm_results_dir, 'kernel_width_vs_psnr.png'))
                gt_mat, lr_mat, sparse_mat, lr_index_mat = load_data_from_swinir_h5(
                    h5_file_path, sample_info['wind_group'], sample_info['source_group'], sample_info['time_step'], scale_factor
                )
                if gt_mat is not None:
                    rco = best_kernel_width * 3
                    reconstruct_mat = get_gaussian_kdm_matrix(
                        measure_mat=sparse_mat,
                        mat_real_size=mat_real_size,
                        sample_index_mat=lr_index_mat,
                        sample_conc_mat=lr_mat,
                        Rco=rco,
                        Gama=best_kernel_width
                    )
                    gkdm_flow(gt_mat, lr_mat, sparse_mat, reconstruct_mat, rco_value=rco,
                              save_path=os.path.join(gkdm_results_dir, 'gkdm_result.png'))


if __name__ == '__main__':
    main()
