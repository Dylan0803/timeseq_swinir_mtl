import argparse
import random
from typing import Dict, Any, Iterable
import os

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib

from datasets.h5_dataset import generate_train_valid_test_dataset

# 设置 matplotlib 后端（Colab 友好）
matplotlib.use('Agg')  # 非交互式后端，适合保存图片
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 使用英文标签以避免中文字体问题


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    """打印单个张量的形状、dtype、最小值和最大值。"""
    try:
        tensor_min = tensor.min().item() if tensor.numel() > 0 else float("nan")
        tensor_max = tensor.max().item() if tensor.numel() > 0 else float("nan")
    except Exception as e:  # 极端情况下的保护
        print(f"[{name}] 计算 min/max 出错: {e}")
        tensor_min, tensor_max = float("nan"), float("nan")

    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={tensor_min:.6f}, max={tensor_max:.6f}"
    )


def print_sample(sample: Dict[str, Any], title: str) -> None:
    """打印样本中每个 key 的基本统计信息。"""
    print(f"\n===== {title} =====")
    for key, val in sample.items():
        if torch.is_tensor(val):
            describe_tensor(key, val)
        else:
            print(f"{key}: 非张量类型，值={val}")


def extract_sim_groups(dataset) -> Iterable:
    """
    提取 dataset 覆盖到的 (wind_group, source_group) 集合。
    通过 index_list -> data_indices 获取映射，确保与 DataLoader 使用一致。
    """
    sim_groups = set()
    for idx_in_list in dataset.index_list:
        info = dataset.data_indices[idx_in_list]
        sim_groups.add((info["wind_group"], info["source_group"]))
    return sim_groups


def visualize_dataset_split(train_size, valid_size, test_size, output_dir="."):
    """可视化数据集划分情况"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 饼图
    sizes = [train_size, valid_size, test_size]
    labels = ['Train', 'Valid', 'Test']
    colors = ['#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.05, 0.05, 0.05)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title('Dataset Split Ratio', fontsize=14, fontweight='bold')

    # 柱状图
    bars = ax2.bar(labels, sizes, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title('Dataset Sample Count', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 在柱状图上添加数值标签
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{size:,}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'dataset_split.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"? 数据集划分图已保存: {output_path}")


def visualize_samples(samples, dataset_type, K, output_dir="."):
    """可视化样本数据（LR序列和HR图像）"""
    num_samples = len(samples)
    if num_samples == 0:
        return

    # 确定是时序数据集还是单帧数据集
    is_seq = 'lr_seq' in samples[0]

    if is_seq:
        # 时序数据集可视化
        fig = plt.figure(figsize=(16, 4 * num_samples))

        for i, sample in enumerate(samples):
            lr_seq = sample['lr_seq'].numpy()  # (K, 1, h, w)
            hr_t = sample['hr_t'].numpy()  # (1, h, w)
            hr_tp1 = sample.get('hr_tp1', None)
            if hr_tp1 is not None:
                hr_tp1 = hr_tp1.numpy()  # (1, h, w)

            K_actual = lr_seq.shape[0]

            # 显示 LR 序列（前几帧和后几帧）
            num_lr_show = min(6, K_actual)
            for j in range(num_lr_show):
                ax = plt.subplot(num_samples, num_lr_show + 3,
                                 i * (num_lr_show + 3) + j + 1)
                lr_frame = lr_seq[j, 0] if len(
                    lr_seq.shape) == 4 else lr_seq[j]
                im = ax.imshow(lr_frame, cmap='viridis', aspect='auto')
                ax.set_title(f'LR_{j+1}', fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

            # 显示 HR_t
            ax = plt.subplot(num_samples, num_lr_show + 3, i *
                             (num_lr_show + 3) + num_lr_show + 1)
            hr_t_frame = hr_t[0] if len(hr_t.shape) == 3 else hr_t
            im = ax.imshow(hr_t_frame, cmap='viridis', aspect='auto')
            ax.set_title('HR_t', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            # 显示 HR_{t+1}（如果存在）
            if hr_tp1 is not None:
                ax = plt.subplot(num_samples, num_lr_show + 3,
                                 i * (num_lr_show + 3) + num_lr_show + 2)
                hr_tp1_frame = hr_tp1[0] if len(hr_tp1.shape) == 3 else hr_tp1
                im = ax.imshow(hr_tp1_frame, cmap='viridis', aspect='auto')
                ax.set_title('HR_{t+1}', fontsize=10, fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

            # 添加样本信息文本
            if 'source_pos' in sample:
                source_pos = sample['source_pos'].numpy()
                info_text = f"Sample {i+1}\nSource: ({source_pos[0]:.3f}, {source_pos[1]:.3f})"
                if 'wind_vector' in sample:
                    wind = sample['wind_vector'].numpy()
                    # wind_vector 可能是 2D 数组 (N, 2)，取平均值或显示形状
                    if wind.ndim == 2 and wind.shape[0] > 2:
                        # 如果是空间场数据，显示平均值和形状
                        wind_mean = wind.mean(axis=0)
                        info_text += f"\nWind: mean=({wind_mean[0]:.3f}, {wind_mean[1]:.3f}), shape={wind.shape}"
                    elif wind.ndim == 1 and wind.shape[0] == 2:
                        # 如果是简单的 2D 向量
                        info_text += f"\nWind: ({wind[0]:.3f}, {wind[1]:.3f})"
                    else:
                        # 其他情况，只显示形状
                        info_text += f"\nWind: shape={wind.shape}"
                plt.figtext(0.02, 0.98 - i * 0.25, info_text, fontsize=9,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'{dataset_type} Dataset Samples Visualization (K={K_actual})',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0.02, 0, 1, 0.98])
        output_path = os.path.join(
            output_dir, f'{dataset_type.lower()}_samples.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"? {dataset_type} 样本可视化已保存: {output_path}")
    else:
        # 单帧数据集可视化
        fig = plt.figure(figsize=(12, 4 * num_samples))

        for i, sample in enumerate(samples):
            lr = sample['lr'].numpy()  # (1, h, w)
            hr = sample['hr'].numpy()  # (1, h, w)

            # LR
            ax = plt.subplot(num_samples, 2, i * 2 + 1)
            lr_frame = lr[0] if len(lr.shape) == 3 else lr
            im = ax.imshow(lr_frame, cmap='viridis', aspect='auto')
            ax.set_title('LR', fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            # HR
            ax = plt.subplot(num_samples, 2, i * 2 + 2)
            hr_frame = hr[0] if len(hr.shape) == 3 else hr
            im = ax.imshow(hr_frame, cmap='viridis', aspect='auto')
            ax.set_title('HR', fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle(f'{dataset_type} Dataset Samples Visualization',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = os.path.join(
            output_dir, f'{dataset_type.lower()}_samples.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"? {dataset_type} 样本可视化已保存: {output_path}")


def visualize_leakage_check(train_groups, val_groups, test_groups,
                            inter_train_val, inter_train_test, inter_val_test,
                            output_dir="."):
    """可视化防泄漏检查结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左侧：集合大小和交集大小
    categories = ['Train', 'Valid', 'Test']
    group_sizes = [len(train_groups), len(val_groups), len(test_groups)]
    colors_bar = ['#66b3ff', '#99ff99', '#ffcc99']

    bars = ax1.bar(categories, group_sizes, color=colors_bar, alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Simulation Group Count', fontsize=12)
    ax1.set_title('Simulation Groups per Dataset',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar, size in zip(bars, group_sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{size}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 右侧：交集大小（应该都是0）
    intersection_labels = ['train∩val', 'train∩test', 'val∩test']
    intersection_sizes = [len(inter_train_val), len(
        inter_train_test), len(inter_val_test)]
    intersection_colors = ['red' if s >
                           0 else 'green' for s in intersection_sizes]

    bars2 = ax2.bar(intersection_labels, intersection_sizes, color=intersection_colors,
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Intersection Size', fontsize=12)
    ax2.set_title('Dataset Intersection Check (should be 0)',
                  fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)

    for bar, size in zip(bars2, intersection_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{size}',
                 ha='center', va='bottom' if size > 0 else 'top',
                 fontsize=11, fontweight='bold', color='white' if size > 0 else 'black')

    # 添加状态文本
    if all(s == 0 for s in intersection_sizes):
        status_text = "? No Leakage"
        status_color = 'green'
    else:
        status_text = "?? Leakage Detected"
        status_color = 'red'

    fig.text(0.5, 0.02, status_text, ha='center', fontsize=16,
             fontweight='bold', color=status_color,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = os.path.join(output_dir, 'leakage_check.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"? 防泄漏检查图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="检查 H5 数据集的时序/单帧数据完整性与防泄漏情况"
    )
    parser.add_argument("--h5", required=True, help="H5 文件路径")
    parser.add_argument("--K", type=int, default=6, help="历史帧数 K")
    parser.add_argument("--bs", type=int, default=4,
                        help="DataLoader batch size")
    parser.add_argument("--num_workers", type=int,
                        default=0, help="DataLoader num_workers")
    parser.add_argument(
        "--use_seq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用时序数据集 MultiTaskSeqDataset（默认 True，可用 --no-use_seq 关闭）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="可视化图片输出目录（默认当前目录）",
    )
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 构建数据集
    print("=== 加载数据集 ===")
    train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
        args.h5,
        K=args.K,
        use_seq_dataset=args.use_seq,
    )

    train_size = len(train_dataset)
    valid_size = len(valid_dataset)
    test_size = len(test_dataset)

    print(f"Train size: {train_size}")
    print(f"Valid size: {valid_size}")
    print(f"Test size:  {test_size}")

    # 可视化数据集划分
    print("\n=== 生成数据集划分可视化 ===")
    visualize_dataset_split(train_size, valid_size, test_size, args.output_dir)

    # 2) 随机抽取 train_dataset 的样本并打印统计
    num_samples = min(2, len(train_dataset))
    if num_samples == 0:
        print("Train 集为空，跳过样本检查。")
    else:
        print("\n=== 随机样本检查 ===")
        indices = random.sample(range(len(train_dataset)), k=num_samples)
        for i, sample_idx in enumerate(indices):
            # 取 data_info 便于在异常时打印
            idx_in_file = train_dataset.index_list[sample_idx]
            data_info = train_dataset.data_indices[idx_in_file]
            try:
                sample = train_dataset[sample_idx]
            except KeyError as e:
                print(
                    f"KeyError: {e}\n"
                    f"  wind={data_info.get('wind_group')}, "
                    f"source={data_info.get('source_group')}, "
                    f"t/time_step={data_info.get('t', data_info.get('time_step'))}"
                )
                raise
            print_sample(sample, f"Train 样本 {i} (idx={sample_idx})")

        # 可视化样本
        print("\n=== 生成样本可视化 ===")
        samples_to_visualize = []
        for sample_idx in indices:
            try:
                sample = train_dataset[sample_idx]
                samples_to_visualize.append(sample)
            except Exception as e:
                print(f"跳过样本 {sample_idx} 的可视化: {e}")
        if samples_to_visualize:
            visualize_samples(samples_to_visualize, "Train",
                              args.K, args.output_dir)

    # 3) DataLoader batch 检查
    if len(train_dataset) > 0:
        print("\n=== DataLoader 批次检查 ===")
        loader = DataLoader(
            train_dataset,
            batch_size=args.bs,
            num_workers=args.num_workers,
            shuffle=False,
        )
        try:
            batch = next(iter(loader))
            print("Batch shapes:")
            for key, val in batch.items():
                if torch.is_tensor(val):
                    print(f"{key}: shape={tuple(val.shape)}, dtype={val.dtype}")
                else:
                    print(f"{key}: 非张量类型，值类型={type(val)}")
        except KeyError as e:
            print("DataLoader 迭代时出现 KeyError，请检查数据键：", e)
            raise
        except StopIteration:
            print("DataLoader 未返回任何批次。")

    # 4) 防泄漏检查：交集必须为 0
    print("\n=== 防泄漏检查 ===")
    train_groups = extract_sim_groups(train_dataset)
    val_groups = extract_sim_groups(valid_dataset)
    test_groups = extract_sim_groups(test_dataset)

    inter_train_val = train_groups & val_groups
    inter_train_test = train_groups & test_groups
    inter_val_test = val_groups & test_groups

    print(f"|train ∩ val|  = {len(inter_train_val)}")
    print(f"|train ∩ test| = {len(inter_train_test)}")
    print(f"|val ∩ test|  = {len(inter_val_test)}")

    if inter_train_val or inter_train_test or inter_val_test:
        print("?? 警告：数据划分存在重叠！")
        if inter_train_val:
            print("  重叠 train∩val:", sorted(inter_train_val))
        if inter_train_test:
            print("  重叠 train∩test:", sorted(inter_train_test))
        if inter_val_test:
            print("  重叠 val∩test:", sorted(inter_val_test))
    else:
        print("? 无泄漏，数据划分互斥。")

    # 可视化防泄漏检查结果
    print("\n=== 生成防泄漏检查可视化 ===")
    visualize_leakage_check(train_groups, val_groups, test_groups,
                            inter_train_val, inter_train_test, inter_val_test,
                            args.output_dir)

    print(f"\n? 所有可视化图片已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
