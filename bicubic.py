# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets.h5_dataset import MultiTaskDataset
import pandas as pd
import h5py
import argparse
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import random
import sys
import json
from argparse import Namespace
from scipy.interpolate import griddata
from scipy.ndimage import zoom, gaussian_filter
from scipy.signal import savgol_filter

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class AdvancedBicubicInterpolation:
    """
    高级双三次插值算法，支持多种可调参数优化
    """

    def __init__(self, upscale_factor=6, **kwargs):
        """
        初始化高级双三次插值器

        参数:
            upscale_factor: 上采样倍数，默认6
            **kwargs: 额外的优化参数
        """
        self.upscale_factor = upscale_factor

        # 插值方法
        # 'linear', 'nearest', 'cubic', 'quintic'
        self.method = kwargs.get('method', 'cubic')

        # 边界处理
        self.fill_value = kwargs.get('fill_value', 0)
        # 'constant', 'reflect', 'wrap'
        self.boundary_mode = kwargs.get('boundary_mode', 'constant')

        # 网格参数
        self.grid_range = kwargs.get('grid_range', [0, 1])  # [最小值, 最大值]
        self.grid_density = kwargs.get(
            'grid_density', 'uniform')  # 'uniform', 'log', 'exp'

        # 后处理参数
        self.clip_range = kwargs.get('clip_range', [0, 1])  # [最小值, 最大值]
        self.smooth_sigma = kwargs.get('smooth_sigma', 0)  # 高斯平滑的sigma值
        self.savgol_window = kwargs.get(
            'savgol_window', None)  # Savitzky-Golay滤波器窗口大小
        self.savgol_polyorder = kwargs.get(
            'savgol_polyorder', 2)  # Savitzky-Golay多项式阶数

        # 边缘增强
        self.edge_enhance = kwargs.get('edge_enhance', False)
        self.edge_strength = kwargs.get('edge_strength', 0.1)

        print(f"初始化高级双三次插值器，参数如下:")
        print(f"  插值方法: {self.method}")
        print(f"  填充值: {self.fill_value}")
        print(f"  网格范围: {self.grid_range}")
        print(f"  平滑sigma: {self.smooth_sigma}")
        print(f"  边缘增强: {self.edge_enhance}")

    def interpolate(self, lr_data):
        """
        使用高级插值算法重建高分辨率气体分布

        参数:
            lr_data: 低分辨率数据 [batch_size, 1, H, W] 或 [1, H, W]

        返回:
            hr_data: 高分辨率数据 [batch_size, 1, H*upscale, W*upscale] 或 [1, H*upscale, W*upscale]
        """
        if len(lr_data.shape) == 4:
            # 批处理模式
            batch_size = lr_data.shape[0]
            hr_results = []

            for i in range(batch_size):
                hr_result = self._single_interpolate(lr_data[i])
                hr_results.append(hr_result)

            return torch.stack(hr_results, dim=0)
        else:
            # 单样本模式
            return self._single_interpolate(lr_data)

    def _single_interpolate(self, lr_data):
        """
        对单个样本进行高级插值

        参数:
            lr_data: 低分辨率数据 [1, H, W]

        返回:
            hr_data: 高分辨率数据 [1, H*upscale, W*upscale]
        """
        # 转换为numpy数组并移除通道维度
        lr_np = lr_data.squeeze().cpu().numpy()

        # 获取原始尺寸
        h, w = lr_np.shape

        # 计算目标尺寸
        target_h = h * self.upscale_factor
        target_w = w * self.upscale_factor

        # 根据密度设置创建网格
        if self.grid_density == 'uniform':
            # 均匀网格
            x_old = np.linspace(self.grid_range[0], self.grid_range[1], w)
            y_old = np.linspace(self.grid_range[0], self.grid_range[1], h)
            x_new = np.linspace(
                self.grid_range[0], self.grid_range[1], target_w)
            y_new = np.linspace(
                self.grid_range[0], self.grid_range[1], target_h)
        elif self.grid_density == 'log':
            # 对数网格，适合指数分布数据
            x_old = np.logspace(
                np.log10(self.grid_range[0] + 1e-6), np.log10(self.grid_range[1]), w)
            y_old = np.logspace(
                np.log10(self.grid_range[0] + 1e-6), np.log10(self.grid_range[1]), h)
            x_new = np.logspace(
                np.log10(self.grid_range[0] + 1e-6), np.log10(self.grid_range[1]), target_w)
            y_new = np.logspace(
                np.log10(self.grid_range[0] + 1e-6), np.log10(self.grid_range[1]), target_h)
        else:  # exponential
            # 指数网格
            x_old = np.exp(np.linspace(
                np.log(self.grid_range[0] + 1e-6), np.log(self.grid_range[1]), w))
            y_old = np.exp(np.linspace(
                np.log(self.grid_range[0] + 1e-6), np.log(self.grid_range[1]), h))
            x_new = np.exp(np.linspace(
                np.log(self.grid_range[0] + 1e-6), np.log(self.grid_range[1]), target_w))
            y_new = np.exp(np.linspace(
                np.log(self.grid_range[0] + 1e-6), np.log(self.grid_range[1]), target_h))

        # 创建网格点
        X_old, Y_old = np.meshgrid(x_old, y_old)
        X_new, Y_new = np.meshgrid(x_new, y_new)

        # 使用scipy griddata进行插值
        points = np.column_stack((X_old.flatten(), Y_old.flatten()))
        values = lr_np.flatten()

        # 插值
        hr_np = griddata(points, values, (X_new, Y_new),
                         method=self.method, fill_value=self.fill_value)

        # 处理NaN值
        if np.any(np.isnan(hr_np)):
            hr_np = np.nan_to_num(hr_np, nan=self.fill_value)

        # 后处理
        hr_np = self._post_process(hr_np)

        # 转换回tensor并添加通道维度
        hr_tensor = torch.from_numpy(hr_np).float().unsqueeze(0)

        return hr_tensor

    def _post_process(self, hr_np):
        """
        对插值结果进行后处理

        参数:
            hr_np: 插值后的numpy数组

        返回:
            处理后的numpy数组
        """
        # 值域限制
        hr_np = np.clip(hr_np, self.clip_range[0], self.clip_range[1])

        # 高斯平滑
        if self.smooth_sigma > 0:
            hr_np = gaussian_filter(hr_np, sigma=self.smooth_sigma)

        # Savitzky-Golay滤波（对1D数据，应用于每行/列）
        if self.savgol_window is not None:
            try:
                # 对行应用滤波
                for i in range(hr_np.shape[0]):
                    hr_np[i, :] = savgol_filter(
                        hr_np[i, :], self.savgol_window, self.savgol_polyorder)
                # 对列应用滤波
                for j in range(hr_np.shape[1]):
                    hr_np[:, j] = savgol_filter(
                        hr_np[:, j], self.savgol_window, self.savgol_polyorder)
            except:
                pass  # 如果窗口大小太大则跳过

        # 边缘增强
        if self.edge_enhance:
            # 使用梯度进行简单的边缘增强
            grad_x = np.gradient(hr_np, axis=1)
            grad_y = np.gradient(hr_np, axis=0)
            edge_strength_map = np.sqrt(grad_x**2 + grad_y**2)
            hr_np = hr_np + self.edge_strength * edge_strength_map
            hr_np = np.clip(hr_np, self.clip_range[0], self.clip_range[1])

        return hr_np


def test_parameter_optimization(data_path, num_samples=2):
    """
    测试不同参数组合的优化效果

    参数:
        data_path: 数据集路径
        num_samples: 测试样本数量
    """
    print(f"\n=== 测试参数优化 ===")

    try:
        # 创建数据集和数据加载器
        dataset = MultiTaskDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

        # 获取一个批次的数据
        batch = next(iter(dataloader))
        lr = batch['lr']
        hr_true = batch['hr']

        # 定义要测试的参数组合
        param_combinations = [
            {
                'name': '基础双三次插值',
                'params': {'method': 'cubic', 'fill_value': 0}
            },
            {
                'name': '线性插值',
                'params': {'method': 'linear', 'fill_value': 0}
            },
            {
                'name': '最近邻插值',
                'params': {'method': 'nearest', 'fill_value': 0}
            },
            {
                'name': '双三次插值+平滑',
                'params': {'method': 'cubic', 'fill_value': 0, 'smooth_sigma': 0.5}
            },
            {
                'name': '双三次插值+边缘增强',
                'params': {'method': 'cubic', 'fill_value': 0, 'edge_enhance': True, 'edge_strength': 0.1}
            },
            {
                'name': '双三次插值+Savitzky-Golay滤波',
                'params': {'method': 'cubic', 'fill_value': 0, 'savgol_window': 5, 'savgol_polyorder': 2}
            },
            {
                'name': '双三次插值+对数网格',
                'params': {'method': 'cubic', 'fill_value': 0, 'grid_density': 'log'}
            }
        ]

        results = []

        for combo in param_combinations:
            print(f"\n--- 测试 {combo['name']} ---")

            # 使用当前参数创建插值器
            interpolator = AdvancedBicubicInterpolation(
                upscale_factor=6, **combo['params'])

            # 执行插值
            hr_pred = interpolator.interpolate(lr)

            # 计算每个样本的指标
            sample_metrics = []
            for i in range(lr.size(0)):
                pred_np = hr_pred[i].squeeze().numpy()
                true_np = hr_true[i].squeeze().numpy()

                # PSNR
                mse = np.mean((pred_np - true_np) ** 2)
                if mse > 0:
                    psnr_val = 20 * np.log10(1.0 / np.sqrt(mse))
                else:
                    psnr_val = float('inf')

                # SSIM
                ssim_val = ssim(pred_np, true_np, data_range=1.0)

                sample_metrics.append({
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'mse': mse
                })

            # 计算平均指标
            avg_psnr = np.mean([m['psnr'] for m in sample_metrics])
            avg_ssim = np.mean([m['ssim'] for m in sample_metrics])
            avg_mse = np.mean([m['mse'] for m in sample_metrics])

            print(f"平均PSNR: {avg_psnr:.2f} dB")
            print(f"平均SSIM: {avg_ssim:.4f}")
            print(f"平均MSE: {avg_mse:.6f}")

            results.append({
                'name': combo['name'],
                'params': combo['params'],
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'avg_mse': avg_mse
            })

        # 找到最佳参数
        best_psnr = max(results, key=lambda x: x['avg_psnr'])
        best_ssim = max(results, key=lambda x: x['avg_ssim'])

        print(f"\n=== 优化结果 ===")
        print(f"最佳PSNR: {best_psnr['name']} ({best_psnr['avg_psnr']:.2f} dB)")
        print(f"最佳SSIM: {best_ssim['name']} ({best_ssim['avg_ssim']:.4f})")

        # 保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv('parameter_optimization_results.csv', index=False)
        print("详细结果已保存到 'parameter_optimization_results.csv'")

        return results

    except Exception as e:
        print(f"参数优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试数据集加载和高级双三次插值')

    parser.add_argument('--data_path', type=str,
                        required=True, help='数据集路径 (.h5文件)')
    parser.add_argument('--num_samples', type=int, default=3, help='测试样本数量')
    parser.add_argument('--no_plots', action='store_true', help='禁用保存可视化图片')
    parser.add_argument('--optimize', action='store_true', help='运行参数优化测试')
    parser.add_argument('--save_dir', type=str,
                        default='results', help='结果保存根目录')

    # 测试模式与样本选择（对齐 evaluate_multi.py）
    parser.add_argument('--test_mode', type=str, default='generalization',
                        choices=['generalization', 'test_set',
                                 'all_generalization', 'all_test_set'],
                        help='测试模式：generalization（泛化测试）、test_set（测试集）、all_generalization（泛化全量）、all_test_set（测试集全量）')
    parser.add_argument('--sample_specs', type=str, default=None,
                        help='泛化测试的样本规格，用分号分隔，例如：wind1_0,s1,50;wind2_0,s2,30')
    parser.add_argument('--test_indices', type=str, default=None,
                        help='测试集索引，用逗号分隔，例如：1,2,3,4,5')

    # 高级插值参数
    parser.add_argument('--method', type=str, default='cubic',
                        choices=['linear', 'nearest', 'cubic', 'quintic'], help='插值方法')
    parser.add_argument('--smooth_sigma', type=float,
                        default=0, help='高斯平滑sigma值')
    parser.add_argument('--edge_enhance', action='store_true', help='启用边缘增强')
    parser.add_argument('--edge_strength', type=float,
                        default=0.1, help='边缘增强强度')
    parser.add_argument('--savgol_window', type=int,
                        default=None, help='Savitzky-Golay滤波器窗口大小')

    args = parser.parse_args()
    return args


def get_dataset_indices(sample_specs, dataset):
    """
    Parse sample specs to dataset indices.
    Format: 'wind_group,source_group,time_steps' separated by ';'
    e.g. 'wind1_0,s1,50;wind2_0,s2,30'
    """
    if not sample_specs:
        return []
    specs = [spec.strip() for spec in sample_specs.split(';') if spec.strip()]
    indices = []
    for spec in specs:
        parts = [p.strip() for p in spec.split(',')]
        if len(parts) < 3:
            continue
        wind_name, source_name, time_steps = parts[0], parts[1], int(parts[2])
        for i, data_info in enumerate(dataset.data_indices):
            if (data_info['wind_group'] == wind_name and
                data_info['source_group'] == source_name and
                    data_info['time_step'] == time_steps):
                indices.append(i)
    return indices


def get_test_set_indices(test_indices, dataset_len):
    """
    Parse comma-separated indices string into a valid index list.
    e.g. '1,2,3,4,5'
    """
    if not test_indices:
        return []
    out = []
    for s in test_indices.split(','):
        s = s.strip()
        if not s:
            continue
        try:
            v = int(s)
            if 0 <= v < dataset_len:
                out.append(v)
        except:
            continue
    return out


def main():
    args = parse_args()
    print("Starting advanced bicubic interpolation test...")
    print("=" * 50)
    save_plots = not args.no_plots

    # Create base save directory and bicubic results directory
    base_save_dir = os.path.abspath(args.save_dir)
    bicubic_root_dir = os.path.join(base_save_dir, 'bicubic_results')
    os.makedirs(bicubic_root_dir, exist_ok=True)

    # Build dataset and subset indices by mode
    from datasets.h5_dataset import MultiTaskDataset
    dataset = MultiTaskDataset(args.data_path)

    if args.test_mode == 'generalization':
        if args.sample_specs is None:
            print("Error: --sample_specs required for generalization mode")
            return
        indices_to_evaluate = get_dataset_indices(args.sample_specs, dataset)
        print(f"Generalization mode, sample_specs: {args.sample_specs}")
    elif args.test_mode == 'all_generalization':
        indices_to_evaluate = list(range(len(dataset)))
        print("All-generalization mode, evaluating all samples")
    elif args.test_mode == 'all_test_set':
        # If you have a real split, replace with that; here we use all as a placeholder.
        indices_to_evaluate = list(range(len(dataset)))
        print("All-test-set mode (treated as full dataset here)")
    else:  # 'test_set'
        if args.test_indices is None:
            print("Error: --test_indices required for test_set mode")
            return
        indices_to_evaluate = get_test_set_indices(
            args.test_indices, len(dataset))
        print(f"Test-set mode, indices: {args.test_indices}")

    if not indices_to_evaluate:
        print("No samples found to evaluate!")
        return

    print(f"Found {len(indices_to_evaluate)} samples to evaluate")

    # Build subset dataset and evaluate on it
    subset_dataset = MultiTaskDataset(
        args.data_path, index_list=indices_to_evaluate, shuffle=False)

    if args.optimize:
        # Optional: adapt test_parameter_optimization to take 'dataset=subset_dataset' if needed.
        test_parameter_optimization(
            args.data_path, num_samples=min(args.num_samples, 2))
    else:
        # Evaluate strictly on the subset in a single pass
        test_advanced_bicubic_interpolation(
            args.data_path,
            num_samples=len(indices_to_evaluate),
            save_plots=save_plots,
            args=args,
            dataset=subset_dataset
        )


def test_advanced_bicubic_interpolation(data_path, num_samples=2, save_plots=True, args=None, dataset=None):
    """
    If 'dataset' is provided, use it directly (subset) for evaluation.
    All figure titles are in English to avoid font warnings.
    """
    print(f"\n=== Testing Advanced Bicubic Interpolation ===")

    try:
        if dataset is None:
            from datasets.h5_dataset import MultiTaskDataset
            dataset = MultiTaskDataset(data_path)

        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

        # Build interpolator from args
        params = {}
        if args:
            params.update({
                'method': getattr(args, 'method', 'cubic'),
                'smooth_sigma': getattr(args, 'smooth_sigma', 0.0),
                'edge_enhance': getattr(args, 'edge_enhance', False),
                'edge_strength': getattr(args, 'edge_strength', 0.1),
                'savgol_window': getattr(args, 'savgol_window', None),
            })
        interpolator = AdvancedBicubicInterpolation(upscale_factor=6, **params)

        batch = next(iter(dataloader))
        lr = batch['lr']
        hr_true = batch['hr']

        print(f"Input LR shape: {lr.shape}")
        print(f"True HR shape: {hr_true.shape}")

        hr_pred = interpolator.interpolate(lr)
        print(f"Predicted HR shape: {hr_pred.shape}")

        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.metrics import structural_similarity as ssim

        # 累计指标以计算平均值
        psnr_list = []
        ssim_list = []
        mse_list = []

        for i in range(lr.size(0)):
            print(f"\n--- Sample {i+1} evaluation ---")
            pred_np = hr_pred[i].squeeze().numpy()
            true_np = hr_true[i].squeeze().numpy()

            mse = np.mean((pred_np - true_np) ** 2)
            psnr_val = 20 * np.log10(1.0 / np.sqrt(mse)
                                     ) if mse > 0 else float('inf')
            ssim_val = ssim(pred_np, true_np, data_range=1.0)

            print(f"PSNR: {psnr_val:.2f} dB")
            print(f"SSIM: {ssim_val:.4f}")
            print(f"MSE: {mse:.6f}")

            # 记录指标
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            mse_list.append(mse)

            if save_plots:
                # Prepare directories: <save_dir>/bicubic_results/sample_{i+1}
                base_save_dir = os.path.abspath(
                    getattr(args, 'save_dir', 'results'))
                bicubic_root_dir = os.path.join(
                    base_save_dir, 'bicubic_results')
                sample_dir = os.path.join(bicubic_root_dir, f'sample_{i+1}')
                os.makedirs(sample_dir, exist_ok=True)

                # Save LR image (no axes, no colorbar)
                plt.figure(figsize=(5, 5))
                lr_np = lr[i].squeeze().numpy()
                plt.imshow(lr_np, cmap='viridis')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(
                    os.path.join(sample_dir, f'advanced_bicubic_sample_{i+1}_LR.png'), dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

                # Save HR image (no axes, no colorbar)
                plt.figure(figsize=(5, 5))
                hr_np = hr_true[i].squeeze().numpy()
                plt.imshow(hr_np, cmap='viridis')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(
                    os.path.join(sample_dir, f'advanced_bicubic_sample_{i+1}_HR.png'), dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

                # Save Bicubic prediction image (no axes, no colorbar)
                plt.figure(figsize=(5, 5))
                plt.imshow(pred_np, cmap='viridis')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(
                    os.path.join(sample_dir, f'advanced_bicubic_sample_{i+1}_Bicubic.png'), dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

                # Save CSVs for LR, HR, Bicubic
                np.savetxt(os.path.join(
                    sample_dir, f'advanced_bicubic_sample_{i+1}_LR.csv'), lr_np, delimiter=',', fmt='%.6f')
                np.savetxt(os.path.join(
                    sample_dir, f'advanced_bicubic_sample_{i+1}_HR.csv'), hr_np, delimiter=',', fmt='%.6f')
                np.savetxt(os.path.join(
                    sample_dir, f'advanced_bicubic_sample_{i+1}_Bicubic.csv'), pred_np, delimiter=',', fmt='%.6f')

                # Save a big composite figure (LR | HR | Bicubic), no axes/colorbars
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.imshow(lr_np, cmap='viridis')
                ax1.axis('off')
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.imshow(hr_np, cmap='viridis')
                ax2.axis('off')
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.imshow(pred_np, cmap='viridis')
                ax3.axis('off')
                plt.tight_layout(pad=0.1)
                plt.savefig(os.path.join(
                    sample_dir, f'advanced_bicubic_sample_{i+1}_COMPOSITE.png'), dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                print(
                    f"Saved images to {sample_dir}")

        # 在多样本评估结束后输出平均指标
        if len(psnr_list) > 0:
            avg_psnr = float(np.mean(psnr_list))
            avg_ssim = float(np.mean(ssim_list))
            avg_mse = float(np.mean(mse_list))

            print("\n=== Average metrics over samples ===")
            print(f"Average PSNR: {avg_psnr:.2f} dB")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print(f"Average MSE: {avg_mse:.6f}")

        return True

    except Exception as e:
        print(f"Advanced bicubic interpolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    main()
