"""
批量可视化脚本：使用 Dataset 加载方式，可视化数据集中的所有样本
优化版本：提高运行速度
"""
from datasets.h5_dataset import MultiTaskDataset, MultiTaskSeqDataset
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，提高速度


def visualize_sample(sample, idx, save_dir, use_seq=False,
                     show_lr=True, show_colorbar=False, dpi=75):
    """
    可视化单个样本（优化版本）

    参数:
        sample: 数据集返回的样本字典
        idx: 样本索引
        save_dir: 保存目录
        use_seq: 是否时序模式
        show_lr: 是否显示 LR 图像
        show_colorbar: 是否显示颜色条
        dpi: 图片分辨率（降低可提高速度）
    """
    # 获取图像和坐标
    if use_seq:
        lr = sample['lr_seq'][-1]  # 最后一帧 LR
        hr = sample['hr_t']
        source_pos_norm = sample['source_pos']
    else:
        lr = sample['lr']
        hr = sample['hr']
        source_pos_norm = sample['source_pos']

    # 转换为 numpy（一次性转换，避免重复）
    if isinstance(hr, torch.Tensor):
        hr = hr.squeeze().cpu().numpy()
    if isinstance(lr, torch.Tensor):
        lr = lr.squeeze().cpu().numpy()
    if isinstance(source_pos_norm, torch.Tensor):
        source_pos_norm = source_pos_norm.cpu().numpy()

    # 反归一化坐标
    H, W = hr.shape
    source_pos_pix = source_pos_norm * (W - 1)

    # 计算 HR 最大值位置（使用更快的 argmax）
    hr_max_idx = hr.argmax()
    hr_max_row, hr_max_col = np.unravel_index(hr_max_idx, hr.shape)

    # 计算距离（简化计算）
    dist_to_max = np.sqrt((source_pos_pix[0] - hr_max_col)**2 +
                          (source_pos_pix[1] - hr_max_row)**2)

    # 获取元信息
    wind_group = sample.get('wind_group', 'unknown')
    source_group = sample.get('source_group', 'unknown')
    time_info = sample.get('t', sample.get('time_step', 'unknown'))

    # 画图（根据 show_lr 决定子图数量）
    if show_lr:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        axes = [None, ax]

    # 左图：LR（如果启用）
    if show_lr:
        ax = axes[0]
        im = ax.imshow(lr, cmap='viridis', origin='upper', vmin=0, vmax=1)
        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(
            f'LR\n{wind_group}/{source_group}, t={time_info}', fontsize=10)
        ax.axis('off')

    # 右图：HR with points
    ax = axes[1]
    im = ax.imshow(hr, cmap='viridis', origin='upper', vmin=0, vmax=1)
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 画红点：source_pos (x, y) 方式
    ax.plot(source_pos_pix[0], source_pos_pix[1], 'r*', markersize=15,
            label=f'Source ({source_pos_pix[0]:.0f},{source_pos_pix[1]:.0f})')

    # 画绿点：HR 最大值位置
    ax.plot(hr_max_col, hr_max_row, 'g*', markersize=15,
            label=f'Max ({hr_max_col},{hr_max_row})')

    ax.set_title(f'Sample {idx} | {wind_group}/{source_group} t={time_info} | d={dist_to_max:.0f}px',
                 fontsize=9)
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    # 保存文件（降低 DPI 提高速度）
    filename = f'sample_{idx:05d}_{wind_group}_{source_group}_t{time_info}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close('all')  # 完全关闭，释放内存


def visualize_all_samples(data_path, save_dir, use_seq=False, K=6,
                          max_samples=None, start_idx=0,
                          show_lr=True, show_colorbar=False, dpi=75):
    """
    可视化数据集中的所有样本（优化版本）

    参数:
        data_path: H5 文件路径
        save_dir: 保存目录
        use_seq: 是否使用时序数据集
        K: 历史帧数（时序数据集）
        max_samples: 最多可视化多少个样本（None 表示全部）
        start_idx: 起始索引
        show_lr: 是否显示 LR 图像（False 可提高速度）
        show_colorbar: 是否显示颜色条（False 可提高速度）
        dpi: 图片分辨率（降低可提高速度，默认75）
    """
    print(f"读取数据集: {data_path}")
    print(f"模式: {'时序' if use_seq else '单帧'}")
    if use_seq:
        print(f"历史帧数 K: {K}")
    print(f"保存目录: {save_dir}")
    print(f"优化设置: show_lr={show_lr}, show_colorbar={show_colorbar}, dpi={dpi}")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建数据集
    if use_seq:
        dataset = MultiTaskSeqDataset(
            data_path, index_list=None, K=K, shuffle=False)
    else:
        dataset = MultiTaskDataset(data_path, index_list=None, shuffle=False)

    total_samples = len(dataset)
    print(f"数据集大小: {total_samples} 个样本")

    # 确定要可视化的样本范围
    end_idx = total_samples
    if max_samples is not None:
        end_idx = min(start_idx + max_samples, total_samples)

    print(f"将可视化样本 {start_idx} 到 {end_idx-1} (共 {end_idx - start_idx} 个)")

    # 遍历样本并可视化
    for idx in tqdm(range(start_idx, end_idx), desc="Visualizing"):
        try:
            sample = dataset[idx]
            visualize_sample(sample, idx, save_dir, use_seq,
                             show_lr, show_colorbar, dpi)
        except Exception as e:
            print(f"\n错误：处理样本 {idx} 时发生错误: {str(e)}")
            continue

    print(f"\n完成！所有可视化图片已保存到: {save_dir}")
    print(f"共处理 {end_idx - start_idx} 个样本")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize all samples in dataset (optimized for speed)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to H5 dataset file")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save visualization images")
    parser.add_argument("--use_seq", action="store_true",
                        help="Use sequential dataset")
    parser.add_argument("--K", type=int, default=6,
                        help="History length K (for seq dataset)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to visualize (None for all)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index (for resuming)")
    parser.add_argument("--no_lr", action="store_true",
                        help="Skip LR visualization (faster)")
    parser.add_argument("--no_colorbar", action="store_true",
                        help="Skip colorbar (faster)")
    parser.add_argument("--dpi", type=int, default=75,
                        help="Image DPI (lower = faster, default 75)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_all_samples(
        args.data_path,
        args.save_dir,
        args.use_seq,
        args.K,
        args.max_samples,
        args.start_idx,
        show_lr=not args.no_lr,
        show_colorbar=not args.no_colorbar,
        dpi=args.dpi
    )
