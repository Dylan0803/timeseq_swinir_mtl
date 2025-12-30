# -*- coding: utf-8 -*-
from models.network_swinir_multi_gdm import SwinIRMulti as SwinIRMultiGDM
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.network_swinir_multi import SwinIRMulti
from models.network_swinir_multi_enhanced import SwinIRMultiEnhanced
from models.network_swinir_hybrid import SwinIRHybrid
from models.network_swinir_fuse import SwinIRFuse
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
import torch.serialization
from argparse import Namespace
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def evaluate_model(model, dataloader, device):
    """
    评估模型性能

    参数:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备

    返回:
        dict: 包含各项评估指标的字典
    """
    model.eval()
    metrics = {
        'gdm_psnr': [],
        'gdm_ssim': [],
        'gsl_position_error': [],
        'gsl_max_pos_error': []
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                # 数据预处理
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                source_pos = batch['source_pos'].to(device)
                hr_max_pos = batch['hr_max_pos'].to(device)

                # 模型推理
                gdm_out, gsl_out = model(lr)

                # 计算GDM指标
                for i in range(gdm_out.size(0)):
                    # PSNR
                    mse = torch.mean((gdm_out[i] - hr[i]) ** 2)
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                    metrics['gdm_psnr'].append(psnr.item())

                    # SSIM (�򻯰�)
                    c1 = (0.01 * 1.0) ** 2
                    c2 = (0.03 * 1.0) ** 2
                    mu1 = torch.mean(gdm_out[i])
                    mu2 = torch.mean(hr[i])
                    sigma1 = torch.var(gdm_out[i])
                    sigma2 = torch.var(hr[i])
                    sigma12 = torch.mean((gdm_out[i] - mu1) * (hr[i] - mu2))
                    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
                    metrics['gdm_ssim'].append(ssim.item())

                # 计算GSL指标
                # 转换到实际坐标（乘以95.0）
                true_pos = source_pos * 95.0
                pred_pos = gsl_out * 95.0

                # 计算反归一化后的距离并除以10
                position_error = torch.sqrt(
                    torch.sum((pred_pos - true_pos) ** 2, dim=1)) / 10.0
                max_pos_error = torch.sqrt(
                    torch.sum((pred_pos - hr_max_pos * 95.0) ** 2, dim=1)) / 10.0

                metrics['gsl_position_error'].extend(
                    position_error.cpu().numpy())
                metrics['gsl_max_pos_error'].extend(
                    max_pos_error.cpu().numpy())

            except KeyError as e:
                print(f"跳过无效数据: {e}")
                continue

    # 计算平均指标
    results = {
        'GDM_PSNR': np.mean(metrics['gdm_psnr']) if metrics['gdm_psnr'] else 0,
        'GDM_SSIM': np.mean(metrics['gdm_ssim']) if metrics['gdm_ssim'] else 0,
        'GSL_Position_Error': np.mean(metrics['gsl_position_error']) if metrics['gsl_position_error'] else 0,
        'GSL_MaxPos_Error': np.mean(metrics['gsl_max_pos_error']) if metrics['gsl_max_pos_error'] else 0
    }

    return results, metrics


def plot_metrics(metrics, save_dir):
    """绘制各项指标分布图"""
    plt.figure(figsize=(15, 10))

    # GDM PSNR分布
    plt.subplot(221)
    plt.hist(metrics['gdm_psnr'], bins=50)
    plt.title('GDM PSNR Distribution')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Count')

    # GDM SSIM分布
    plt.subplot(222)
    plt.hist(metrics['gdm_ssim'], bins=50)
    plt.title('GDM SSIM Distribution')
    plt.xlabel('SSIM')
    plt.ylabel('Count')

    # GSL位置误差分布
    plt.subplot(223)
    plt.hist(metrics['gsl_position_error'], bins=50)
    plt.title('GSL Position Error Distribution')
    plt.xlabel('Error (normalized)')
    plt.ylabel('Count')

    # GSL最大值位置误差分布
    plt.subplot(224)
    plt.hist(metrics['gsl_max_pos_error'], bins=50)
    plt.title('GSL Max Position Error Distribution')
    plt.xlabel('Error (normalized)')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_metrics.png'), dpi=120)
    plt.close()


def visualize_results(lr, hr, gdm_out, gsl_out, source_pos, hr_max_pos, save_path):
    """可视化推理结果"""
    # 转换为numpy数组并移除batch维度
    lr = lr.squeeze().cpu().numpy()
    hr = hr.squeeze().cpu().numpy()
    gdm_out = gdm_out.squeeze().cpu().numpy()

    # 计算差值图 - 显示非归一化值
    diff = hr - gdm_out

    # ����ͼ�񣬵�����ͼ��������ɵײ�ͼע
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 将总标题放在底部
    fig.suptitle('Gas Concentration Distribution', fontsize=16, y=0.02)

    # 低分辨率输入
    im0 = axes[0, 0].imshow(lr, cmap='viridis')
    axes[0, 0].set_title('Low Resolution Input', pad=20, y=-0.15)
    axes[0, 0].axis('off')
    # 移除颜色条

    # 高分辨率真实值
    im1 = axes[0, 1].imshow(hr, cmap='viridis')
    axes[0, 1].set_title('High Resolution Ground Truth', pad=20, y=-0.15)
    axes[0, 1].axis('off')
    # 移除颜色条

    # 模型输出超分辨率
    im2 = axes[1, 0].imshow(gdm_out, cmap='viridis')
    axes[1, 0].set_title('Super Resolution Output', pad=20, y=-0.15)
    axes[1, 0].axis('off')
    # 移除颜色条

    # 只在非gdm模型时显示泄漏源
    if (gsl_out is not None) and (source_pos is not None) and (hr_max_pos is not None):
        gsl_out = gsl_out.squeeze().cpu().numpy()
        source_pos = source_pos.squeeze().cpu().numpy()
        hr_max_pos = hr_max_pos.squeeze().cpu().numpy()
        # 转换到实际坐标（乘以95.0）
        true_pos = source_pos * 95.0
        pred_pos = gsl_out * 95.0
        # 标记真实泄漏源位置（红色星形）
        axes[1, 0].plot(true_pos[0], true_pos[1], 'r*',
                        markersize=15, label='True Source')
        # 标记预测泄漏源位置（绿色星形）
        axes[1, 0].plot(pred_pos[0], pred_pos[1], 'g*',
                        markersize=15, label='Predicted Source')
        # 添加图例
        axes[1, 0].legend(loc='upper right')

    # 差值图 - 显示非归一化值，使用RdBu_r颜色映射
    im3 = axes[1, 1].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('SR-HR', pad=20, y=-0.15)
    axes[1, 1].axis('off')
    # 移除颜色条

    # 调整布局，为底部标题留出空间
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Ϊ�ײ�������������ռ�

    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()

    # 只在pred_pos和true_pos都已定义时返回距离
    if ('pred_pos' in locals()) and ('true_pos' in locals()):
        distance = np.sqrt(np.sum((pred_pos - true_pos) ** 2)) / 10.0
        return distance
    else:
        return None


def save_sample_data(lr, hr, gdm_out, gsl_out, source_pos, hr_max_pos, save_dir, sample_idx, model_type):
    """保存每个样本的详细数据到指定目录，用于后续分析"""
    import numpy as np

    # save_dir 是各样本的目录
    os.makedirs(save_dir, exist_ok=True)

    # 转换为numpy数组并移除batch维度
    lr_np = lr.squeeze().cpu().numpy()
    hr_np = hr.squeeze().cpu().numpy()
    gdm_out_np = gdm_out.squeeze().cpu().numpy()

    # 保存LR、HR、GDM输出为CSV
    np.savetxt(os.path.join(save_dir, 'lr.csv'),
               lr_np, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(save_dir, 'hr.csv'),
               hr_np, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(save_dir, 'gdm_out.csv'),
               gdm_out_np, delimiter=',', fmt='%.6f')

    # 保存差值数据为CSV
    diff = hr_np - gdm_out_np
    np.savetxt(os.path.join(save_dir, 'difference.csv'),
               diff, delimiter=',', fmt='%.6f')

    # 保存GSL相关数据（如果存在）
    if model_type != 'swinir_gdm' and gsl_out is not None and source_pos is not None:
        gsl_out_np = gsl_out.squeeze().cpu().numpy()
        source_pos_np = source_pos.squeeze().cpu().numpy()
        hr_max_pos_np = hr_max_pos.squeeze().cpu(
        ).numpy() if hr_max_pos is not None else None

        # 对 original 模型，全部保存为 CSV 格式
        if model_type == 'original':
            np.savetxt(os.path.join(save_dir, 'gsl_out.csv'),
                       gsl_out_np, delimiter=',', fmt='%.6f')
            np.savetxt(os.path.join(save_dir, 'source_pos.csv'),
                       source_pos_np, delimiter=',', fmt='%.6f')
            if hr_max_pos_np is not None:
                np.savetxt(os.path.join(save_dir, 'hr_max_pos.csv'),
                           hr_max_pos_np, delimiter=',', fmt='%.6f')
        else:
            np.save(os.path.join(save_dir, 'gsl_out.npy'), gsl_out_np)
            np.save(os.path.join(save_dir, 'source_pos.npy'), source_pos_np)
            if hr_max_pos_np is not None:
                np.save(os.path.join(save_dir,
                        'hr_max_pos.npy'), hr_max_pos_np)

    # 保存元数据信息
    metadata = {
        'sample_idx': sample_idx,
        'model_type': model_type,
        'lr_shape': lr_np.shape,
        'hr_shape': hr_np.shape,
        'gdm_out_shape': gdm_out_np.shape,
        'diff_shape': diff.shape,
        'lr_range': [float(lr_np.min()), float(lr_np.max())],
        'hr_range': [float(hr_np.min()), float(hr_np.max())],
        'gdm_out_range': [float(gdm_out_np.min()), float(gdm_out_np.max())],
        'diff_range': [float(diff.min()), float(diff.max())]
    }

    # 保存元数据为JSON文件
    import json
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"样本 {sample_idx} 的数据已保存到: {save_dir}")


def infer_model(model, data_path, save_dir, num_samples=5, sample_indices=None, model_type='original'):
    """
    ģ����������

    ����:
        model: ѵ���õ�ģ��
        data_path: ����·��
        save_dir: ��������Ŀ¼
        num_samples: Ҫ��������������
        sample_indices: ָ��Ҫ���������������б������ΪNone�����ѡ��
    """
    # 创建模型专属的结果目录
    base_save_dir = os.path.abspath(save_dir)
    if model_type == 'swinir_gdm':
        model_root = os.path.join(base_save_dir, 'swinir_gdm_results')
    else:  # original ������
        model_root = os.path.join(base_save_dir, 'swinir_multi_results')
    os.makedirs(model_root, exist_ok=True)

    # 创建数据集，设置shuffle=False确保数据不随机
    dataset = MultiTaskDataset(data_path, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 获取设备
    device = next(model.parameters()).device

    # 初始化指标
    total_psnr = 0
    total_position_error = 0
    total_mse = 0
    total_ssim = 0
    valid_samples = 0

    # 如果提供了样本索引，则只评估指定的样本
    if sample_indices is not None and len(sample_indices) > 0:
        indices_to_evaluate = sample_indices
    else:
        indices_to_evaluate = list(range(len(dataset)))
        if num_samples < len(indices_to_evaluate):
            indices_to_evaluate = indices_to_evaluate[:num_samples]

    print(f"要评估的样本: {indices_to_evaluate}")

    # 初始化指标����������
    for idx in indices_to_evaluate:
        try:
            # 获取指定的样本数据
            batch = dataset[idx]

            # 将数据移动到设备上
            lr = batch['lr'].unsqueeze(0).to(device)  # ����batchά��
            hr = batch['hr'].unsqueeze(0).to(device)
            source_pos = batch['source_pos'].unsqueeze(0).to(
                device) if 'source_pos' in batch else None

            # ģ������
            with torch.no_grad():
                if model_type == 'swinir_gdm':
                    gdm_out = model(lr)
                    gsl_out = None
                else:
                    gdm_out, gsl_out = model(lr)

            # 计算HR图像最大值位置
            hr_max_pos = torch.tensor([torch.argmax(hr[0, 0]) % hr.shape[3],
                                       torch.argmax(hr[0, 0]) // hr.shape[3]],
                                      dtype=torch.float32).to(device)
            hr_max_pos = hr_max_pos / torch.tensor([hr.shape[3], hr.shape[2]],
                                                   dtype=torch.float32).to(device)

            # 计算各项指标
            # MSE损失
            mse = F.mse_loss(gdm_out, hr)

            # PSNR
            psnr = 10 * torch.log10(1.0 / mse)

            # SSIM计算
            c1 = (0.01 * 1.0) ** 2
            c2 = (0.03 * 1.0) ** 2
            mu1 = torch.mean(gdm_out)
            mu2 = torch.mean(hr)
            sigma1 = torch.var(gdm_out)
            sigma2 = torch.var(hr)
            sigma12 = torch.mean((gdm_out - mu1) * (hr - mu2))
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))

            total_psnr += psnr.item()
            total_mse += mse.item()
            total_ssim += ssim.item()
            valid_samples += 1

            # 只在非gdm模型时计算GSL误差
            if model_type != 'swinir_gdm':
                # 转换到实际坐标（乘以95.0）
                true_pos = source_pos * 95.0
                pred_pos = gsl_out * 95.0
                # 计算反归一化后的直线距离并除以10（转换为米）
                position_error = torch.sqrt(
                    torch.sum((pred_pos - true_pos) ** 2)) / 10.0
                total_position_error += position_error.item()

            # 为每个样本创建独立目录
            sample_dir = os.path.join(model_root, f'sample_{idx}')
            os.makedirs(sample_dir, exist_ok=True)

            # 保存样本数据（npy和metadata）
            save_sample_data(lr, hr, gdm_out, gsl_out, source_pos, hr_max_pos,
                             sample_dir, idx, model_type)

            # 保存合成图到样本目录
            save_path = os.path.join(sample_dir, f'sample_{idx}_composite.png')
            if model_type == 'swinir_gdm':
                visualize_results(lr, hr, gdm_out, None, None, None, save_path)
            else:
                visualize_results(lr, hr, gdm_out, gsl_out,
                                  source_pos, hr_max_pos, save_path)

            # 额外单独保存 LR / HR / GDM 图像（无标题/颜色条）
            lr_np = lr.squeeze().detach().cpu().numpy()
            hr_np = hr.squeeze().detach().cpu().numpy()
            gdm_np = gdm_out.squeeze().detach().cpu().numpy()

            plt.figure(figsize=(5, 5))
            plt.imshow(lr_np, cmap='viridis')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(
                sample_dir, f'sample_{idx}_LR.png'), dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.figure(figsize=(5, 5))
            plt.imshow(hr_np, cmap='viridis')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(
                sample_dir, f'sample_{idx}_HR.png'), dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.figure(figsize=(5, 5))
            plt.imshow(gdm_np, cmap='viridis')
            # 对 original 模型，在单独GDM图上标注真实/预测泄漏源位置
            if model_type != 'swinir_gdm' and gsl_out is not None and source_pos is not None:
                gsl_out_np = gsl_out.squeeze().detach().cpu().numpy()
                source_pos_np = source_pos.squeeze().detach().cpu().numpy()
                true_pos = source_pos_np * 95.0
                pred_pos = gsl_out_np * 95.0
                plt.plot(true_pos[0], true_pos[1], 'r*',
                         markersize=15, label='True Source')
                plt.plot(pred_pos[0], pred_pos[1], 'g*',
                         markersize=15, label='Predicted Source')
                plt.legend(loc='upper right')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(
                sample_dir, f'sample_{idx}_GDM.png'), dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

        except KeyError as e:
            print(f"������Ч����: {e}")
            continue
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue

    # 计算平均指标
    if valid_samples > 0:
        avg_psnr = total_psnr / valid_samples
        avg_mse = total_mse / valid_samples
        avg_ssim = total_ssim / valid_samples
        print(f"\n�������:")
        print(f"有效样本数: {valid_samples}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        if model_type != 'swinir_gdm':
            avg_position_error = total_position_error / valid_samples
            print(f"Average Position Error: {avg_position_error:.4f} m")
    else:
        print("没有有效样本成功处理")


def get_dataset_indices(sample_specs, dataset):
    """
    ������������ȡ���ݼ��е�ʵ���������ϸ� sample_specs ˳�򷵻�
    ֻ�������ڵ����ݣ����������ڵ���
    """
    actual_indices = []
    if not sample_specs:
        return actual_indices

    # 先构建一个映射表，只包含存在的数据
    index_map = {}
    for idx in range(len(dataset)):
        try:
            data_info = dataset.data_indices[idx]
            key = f"{data_info['wind_group']},{data_info['source_group']},{data_info['time_step']}"
            index_map[key] = idx
        except Exception:
            # 忽略不存在的数据
            continue

    # 按 sample_specs 顺序查找，只返回存在的数据
    for spec in sample_specs:
        try:
            idx = index_map.get(spec)
            if idx is not None:
                actual_indices.append(idx)
                print(f"找到匹配数据: {spec}, 索引={idx}")
            else:
                print(f"未找到匹配数据: {spec}")
        except Exception:
            # 忽略不存在的数据
            continue

    return actual_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SwinIR Multi-task Model')

    # 模型类型选择参数
    parser.add_argument('--model_type', type=str, default='original',
                        choices=['original', 'enhanced',
                                 'hybrid', 'fuse', 'swinir_gdm'],
                        help='选择模型类型: original, enhanced, hybrid, fuse, swinir_gdm')

    # 必需缺失的参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='使用的设备')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='要评估的样本数量')

    # 测试模式选择（包括all_generalization和all_test_set）
    parser.add_argument('--test_mode', type=str, default='generalization',
                        choices=['generalization', 'test_set',
                                 'all_generalization', 'all_test_set'],
                        help='测试模式：generalization（泛化测试），test_set（测试集），all_generalization（泛化全量），all_test_set（测试集全量）')

    # 样本选择参数
    parser.add_argument('--sample_specs', type=str, default=None,
                        help='要测试的样本规格，用分号分隔，如：wind1_0,s1,50;wind2_0,s2,30')
    parser.add_argument('--test_indices', type=str, default=None,
                        help='测试集样本索引，用逗号分隔，如：1,2,3,4,5')

    # 训练 config 参数
    parser.add_argument('--config', type=str, default=None,
                        help='训练参数json路径，如training_args.json')

    parser.add_argument('--upsampler', type=str, default='nearest+conv',
                        choices=['nearest+conv', 'pixelshuffle'],
                        help='上采样方式: nearest+conv（默认）或 pixelshuffle')

    # 新增：仅对 original(SwinIRMulti) 有效的 RSTB 组数参数
    parser.add_argument('--multi_rstb', type=int, default=None,
                        help='仅对 original(SwinIRMulti) 生效：设置 RSTB 组数（如 2/4/6/8/...）')

    args = parser.parse_args()
    return args


def create_model(args):
    """根据参数创建模型"""
    # 首先读取 config（training_args.json）
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        # 只使用模型相关参数
        ignore_keys = [
            'model_type', 'model_path', 'data_path', 'save_dir', 'device', 'num_samples',
            'test_mode', 'sample_specs', 'test_indices', 'seed', 'batch_size', 'num_epochs',
            'lr', 'weight_decay', 'use_test_set', 'upsampler'
        ]
        model_params = {k: v for k,
                        v in config.items() if k not in ignore_keys}
        print("使用config中的模型参数如下（请校验）：")
        for k, v in model_params.items():
            print(f"  {k}: {v}")
        # 选择模型类型
        if args.model_type == 'original':
            model = SwinIRMulti(**model_params)
        elif args.model_type == 'enhanced':
            model = SwinIRMultiEnhanced(**model_params)
        elif args.model_type == 'hybrid':
            model = SwinIRHybrid(**model_params)
        elif args.model_type == 'swinir_gdm':
            model = SwinIRMultiGDM(**model_params)  # 可根据需要调整参数
        else:
            model = SwinIRFuse(**model_params)
        return model

    # 基础模型参数
    base_params = {
        'img_size': 16,  # LR图像大小
        'in_chans': 1,   # 输入通道数
        'upscale': 6,    # 上采样倍率
        'img_range': 1.,  # 图像范围
        'upsampler': args.upsampler  # 可选择的上采样器
    }

    # 原始模型参数：可按 multi_rstb 设置 RSTB 组数
    default_depth_per_group = 6
    default_heads_per_group = 6
    default_groups = 4
    groups = args.multi_rstb if (
        hasattr(args, 'multi_rstb') and args.multi_rstb) else default_groups
    depths = [default_depth_per_group for _ in range(groups)]
    num_heads = [default_heads_per_group for _ in range(groups)]
    original_params = {
        **base_params,
        'window_size': 8,  # Swin Transformer窗口大小
        'depths': depths,  # RSTB 组深度列表
        'embed_dim': 60,  # 嵌入维度
        'num_heads': num_heads,  # 注意力头数列表，与 depths 对齐
        'mlp_ratio': 2.,  # MLP 比例
    }

    # 增强模型参数
    enhanced_params = {
        **base_params,
        'window_size': 8,  # Swin Transformer窗口大小
        'depths': [6, 6, 6, 6],  # Swin Transformer层数
        'embed_dim': 60,  # 嵌入维度
        'num_heads': [6, 6, 6, 6],  # 注意力头数
        'mlp_ratio': 2.,  # MLP 比例
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
        'norm_layer': nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'resi_connection': '1conv'
    }

    # 混合架构模型参数
    hybrid_params = {
        **base_params,
        'window_size': 8,  # Swin Transformer窗口大小
        'depths': [6, 6, 6, 6],  # Swin Transformer层数
        'embed_dim': 60,  # 嵌入维度
        'num_heads': [6, 6, 6, 6],  # 注意力头数
        'mlp_ratio': 2.,  # MLP 比例
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
        'norm_layer': nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'resi_connection': '1conv'
    }

    # Fuse模型参数 (移除了resi_connection)
    fuse_params = {
        'img_size': 16,
        'in_chans': 1,
        'upscale': 6,
        'img_range': 1.,
        'upsampler': args.upsampler,
        'window_size': 8,
        'depths': [6, 6, 6, 6],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
        'norm_layer': nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
    }

    if args.model_type == 'original':
        model = SwinIRMulti(**original_params)
    elif args.model_type == 'enhanced':
        model = SwinIRMultiEnhanced(**enhanced_params)
    elif args.model_type == 'hybrid':
        model = SwinIRHybrid(**hybrid_params)
    elif args.model_type == 'swinir_gdm':
        model = SwinIRMultiGDM(**original_params)  # 可根据需要调整参数
    else:  # fuse
        model = SwinIRFuse(**fuse_params)

    return model


def get_test_set_indices(test_indices_str, dataset):
    """
    ���ݲ��Լ������ַ�����ȡҪ��������������

    ����:
        test_indices_str: ���ŷָ��������ַ��������磺"1,2,3,4,5"
        dataset: ���ݼ�����

    ����:
        list: Ҫ���������������б�
    """
    if not test_indices_str:
        return []

    try:
        # 解析索引字符串
        indices = [int(idx.strip()) for idx in test_indices_str.split(',')]
        # 验证索引是否有效
        valid_indices = [idx for idx in indices if 0 <= idx < len(dataset)]
        if len(valid_indices) != len(indices):
            print(f"警告：部分索引超出范围，已过滤无效索引")
            print(f"有效索引范围：0 到 {len(dataset)-1}")
            print(
                f"��Ч��������{[idx for idx in indices if idx < 0 or idx >= len(dataset)]}")
        return valid_indices
    except ValueError as e:
        print(f"解析无效索引格式 - {e}")
        print(f"����������Χ��0 �� {len(dataset)-1}")
        return []


def batch_infer_model(model, dataset, save_dir, model_type, device='cuda', batch_size=16, num_workers=4, max_visualize=10):
    """
    ��������������������all_generalization��all_test_set
    """
    import torch
    from torch.utils.data import DataLoader
    import os

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    model = model.to(device)
    model.eval()

    total_psnr = 0
    total_position_error = 0
    total_mse = 0
    total_ssim = 0
    valid_samples = 0
    visualized = 0  # 可视化计数器
    total_processed = 0  # 全局处理的样本计数器，用于 sample_{idx}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Batch Evaluating")):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            source_pos = batch.get('source_pos', None)
            if source_pos is not None:
                source_pos = source_pos.to(device)

            if model_type == 'swinir_gdm':
                gdm_out = model(lr)
                gsl_out = None
            else:
                gdm_out, gsl_out = model(lr)

            # 计算指标（批处理）
            mse = F.mse_loss(gdm_out, hr, reduction='none')
            mse = mse.view(mse.size(0), -1).mean(dim=1)
            psnr = 10 * torch.log10(1.0 / mse)
            c1 = (0.01 * 1.0) ** 2
            c2 = (0.03 * 1.0) ** 2
            mu1 = gdm_out.mean(dim=[1, 2, 3])
            mu2 = hr.mean(dim=[1, 2, 3])
            sigma1 = gdm_out.var(dim=[1, 2, 3])
            sigma2 = hr.var(dim=[1, 2, 3])
            sigma12 = ((gdm_out - mu1[:, None, None, None]) *
                       (hr - mu2[:, None, None, None])).mean(dim=[1, 2, 3])
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))

            total_psnr += psnr.sum().item()
            total_mse += mse.sum().item()
            total_ssim += ssim.sum().item()
            valid_samples += lr.size(0)

            # GSL误差
            if model_type != 'swinir_gdm' and source_pos is not None and gsl_out is not None:
                true_pos = source_pos * 95.0
                pred_pos = gsl_out * 95.0
                position_error = torch.sqrt(
                    torch.sum((pred_pos - true_pos) ** 2, dim=1)) / 10.0
                total_position_error += position_error.sum().item()

            # 为批次中的所有样本创建独立目录，保存数据和图片（受max_visualize限制）
            batch_size_now = lr.size(0)
            for j in range(batch_size_now):
                sample_idx_global = total_processed
                sample_dir = os.path.join(
                    save_dir, f'sample_{sample_idx_global}')
                os.makedirs(sample_dir, exist_ok=True)

                # 进行推理���ݣ�CSV/NPY����ģ�����͵ļ����߼���
                save_sample_data(
                    lr[j:j+1], hr[j:j+1], gdm_out[j:j+1],
                    (gsl_out[j:j+1] if (gsl_out is not None and model_type !=
                     'swinir_gdm') else None),
                    (source_pos[j:j+1] if (source_pos is not None and model_type !=
                     'swinir_gdm') else None),
                    None,
                    sample_dir, sample_idx_global, model_type
                )

                # 保存合成图
                composite_path = os.path.join(
                    sample_dir, f'sample_{sample_idx_global}_composite.png')
                if model_type == 'swinir_gdm':
                    visualize_results(
                        lr[j:j+1], hr[j:j+1], gdm_out[j:j+1], None, None, None, composite_path)
                else:
                    visualize_results(
                        lr[j:j+1], hr[j:j+1], gdm_out[j:j+1],
                        (gsl_out[j:j+1] if gsl_out is not None else None),
                        (source_pos[j:j+1]
                         if source_pos is not None else None),
                        None, composite_path
                    )

                # 单独保存 LR / HR / GDM 图
                lr_np = lr[j].squeeze().detach().cpu().numpy()
                hr_np = hr[j].squeeze().detach().cpu().numpy()
                gdm_np = gdm_out[j].squeeze().detach().cpu().numpy()

                plt.figure(figsize=(5, 5))
                plt.imshow(lr_np, cmap='viridis')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(
                    sample_dir, f'sample_{sample_idx_global}_LR.png'), dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.figure(figsize=(5, 5))
                plt.imshow(hr_np, cmap='viridis')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(
                    sample_dir, f'sample_{sample_idx_global}_HR.png'), dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.figure(figsize=(5, 5))
                plt.imshow(gdm_np, cmap='viridis')
                # original 模型在单独GDM图上标注泄漏源真实/预测
                if model_type != 'swinir_gdm' and gsl_out is not None and source_pos is not None:
                    gsl_out_np = gsl_out[j].squeeze().detach().cpu().numpy()
                    source_pos_np = source_pos[j].squeeze(
                    ).detach().cpu().numpy()
                    true_pos = source_pos_np * 95.0
                    pred_pos = gsl_out_np * 95.0
                    plt.plot(true_pos[0], true_pos[1], 'r*',
                             markersize=15, label='True Source')
                    plt.plot(pred_pos[0], pred_pos[1], 'g*',
                             markersize=15, label='Predicted Source')
                    plt.legend(loc='upper right')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(
                    sample_dir, f'sample_{sample_idx_global}_GDM.png'), dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

                total_processed += 1

            # 另外，为批次预览生成合成图（不影响主要功能）
            if visualized < max_visualize:
                preview_path = os.path.join(
                    save_dir, f'batch_{i}_sample_preview.png')
                visualize_results(lr[0:1], hr[0:1], gdm_out[0:1], None if model_type == 'swinir_gdm' else (
                    gsl_out[0:1] if gsl_out is not None else None), None, None, preview_path)
                visualized += 1

    # ���ƽ��ָ��
    if valid_samples > 0:
        avg_psnr = total_psnr / valid_samples
        avg_mse = total_mse / valid_samples
        avg_ssim = total_ssim / valid_samples
        print(f"\n�����������:")
        print(f"有效样本数: {valid_samples}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        if model_type != 'swinir_gdm':
            avg_position_error = total_position_error / valid_samples
            print(f"Average Position Error: {avg_position_error:.4f} m")
    else:
        print("没有有效样本成功处理")


def main():
    args = parse_args()

    # 获取设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型
    model = create_model(args)

    # 创建模型
    print(f"Loading model from {args.model_path}")
    try:
        # 针对 original/enhanced/hybrid 做更健壮的权重加载
        if args.model_type in ['original', 'enhanced', 'hybrid']:
            torch.serialization.add_safe_globals([Namespace])
            checkpoint = torch.load(
                args.model_path, map_location='cpu', weights_only=False)
            # 判断是完整checkpoint还是state_dict
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            # fuse模型直接加载
            torch.serialization.add_safe_globals([Namespace])
            checkpoint = torch.load(
                args.model_path, map_location='cpu', weights_only=False)
            # 检查是否是完整checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        return

    model = model.to(device)
    model.eval()

    use_batch = False  # <--- 增加这一行，默认不开启批处理

    # 加载数据集
    dataset = None
    # 根据测试模式选择要评估的样本
    indices_to_evaluate = []
    if args.test_mode == 'generalization':
        # 泛化测试模式
        if args.sample_specs is not None:
            sample_specs = [spec.strip()
                            for spec in args.sample_specs.split(';')]
            dataset = MultiTaskDataset(args.data_path)
            indices_to_evaluate = get_dataset_indices(sample_specs, dataset)
            print(f"使用泛化测试模式，样本规格：{args.sample_specs}")
        else:
            print("���󣺷�������ģʽ��Ҫ�ṩ sample_specs ����")
            return
    elif args.test_mode == 'all_generalization':
        dataset = MultiTaskDataset(args.data_path)
        indices_to_evaluate = list(range(len(dataset)))
        print("使用泛化测试全量模式，评估所有样本")
        use_batch = True
    elif args.test_mode == 'all_test_set':
        from datasets.h5_dataset import generate_train_valid_test_dataset
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            args.data_path, seed=42)
        dataset = test_dataset
        indices_to_evaluate = list(range(len(dataset)))
        print("使用测试集全量模式，评估所有样本")
        use_batch = True
    else:
        # 测试集模式
        from datasets.h5_dataset import generate_train_valid_test_dataset
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            args.data_path, seed=42)
        dataset = test_dataset
        if args.test_indices is not None:
            indices_to_evaluate = get_test_set_indices(
                args.test_indices, dataset)
            print(f"使用测试集模式，测试索引：{args.test_indices}")
        else:
            print("���󣺲��Լ�ģʽ��Ҫ�ṩ test_indices ����")
            return

    if not indices_to_evaluate:
        print("没有找到要评估的样本！")
        return
    print(f"找到 {len(indices_to_evaluate)} 个要评估的样本")
    # 进行推理
    if use_batch:
        # 创建模型专属的结果目录
        base_save_dir = os.path.abspath(args.save_dir)
        if args.model_type == 'swinir_gdm':
            model_root = os.path.join(base_save_dir, 'swinir_gdm_results')
        else:
            model_root = os.path.join(base_save_dir, 'swinir_multi_results')
        os.makedirs(model_root, exist_ok=True)
        batch_infer_model(model, dataset, model_root,
                          args.model_type, device=args.device)
    else:
        infer_model(model, args.data_path, args.save_dir,
                    args.num_samples, indices_to_evaluate, args.model_type)


if __name__ == '__main__':
    main()
