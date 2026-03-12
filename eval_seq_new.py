"""
时序模型评估脚本 - 专门用于评估 SwinIRMulti 时序模型
支持 SR@t, Pred@t+1, GSL 三个任务的评估
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.network_swinir_multi import SwinIRMulti
from datasets.h5_dataset import generate_train_valid_test_dataset, MultiTaskSeqDataset


def calculate_psnr(img1, img2, data_range=1.0):
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)

    参数:
        img1, img2: torch.Tensor, shape (B, C, H, W) 或 (C, H, W) 或 (H, W)
        data_range: 数据范围，默认 1.0

    返回:
        float: PSNR 值
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return psnr.item()


def calculate_ssim(img1, img2, data_range=1.0):
    """
    计算 SSIM (Structural Similarity Index)
    简化版本：使用全局均值、方差、协方差

    参数:
        img1, img2: torch.Tensor, shape (B, C, H, W) 或 (C, H, W) 或 (H, W)
        data_range: 数据范围，默认 1.0

    返回:
        float: SSIM 值
    """
    # 确保是 4D tensor (B, C, H, W)
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # 展平为 (B, C*H*W)
    img1_flat = img1.view(img1.shape[0], -1)
    img2_flat = img2.view(img2.shape[0], -1)

    # 计算均值和方差
    mu1 = torch.mean(img1_flat, dim=1)
    mu2 = torch.mean(img2_flat, dim=1)

    sigma1_sq = torch.var(img1_flat, dim=1)
    sigma2_sq = torch.var(img2_flat, dim=1)
    sigma12 = torch.mean((img1_flat - mu1.unsqueeze(1)) *
                         (img2_flat - mu2.unsqueeze(1)), dim=1)

    # SSIM 公式
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim.mean().item()


def load_checkpoint(model, checkpoint_path, device):
    """
    加载 checkpoint，兼容多种格式

    参数:
        model: 模型实例
        checkpoint_path: checkpoint 文件路径
        device: 设备

    返回:
        dict: 加载的 checkpoint 信息（如果有）
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 尝试多种可能的 key
    state_dict = None
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            # 假设整个 dict 就是 state_dict
            state_dict = ckpt
    else:
        state_dict = ckpt

    # 使用 strict=False 以兼容缺失的键（如新增的 temporal_fc）
    result = model.load_state_dict(state_dict, strict=False)

    # 打印缺失和意外的键（只显示前10个）
    if result.missing_keys:
        print(f"Missing keys (showing first 10): {result.missing_keys[:10]}")
        if len(result.missing_keys) > 10:
            print(f"... and {len(result.missing_keys) - 10} more missing keys")

    if result.unexpected_keys:
        print(
            f"Unexpected keys (showing first 10): {result.unexpected_keys[:10]}")
        if len(result.unexpected_keys) > 10:
            print(
                f"... and {len(result.unexpected_keys) - 10} more unexpected keys")

    return ckpt if isinstance(ckpt, dict) else None


def evaluate_model(model, test_loader, device, enable_pred=False, max_viz=10, save_dir=None):
    """
    评估模型性能

    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
        enable_pred: 是否评估 Pred@t+1 任务
        max_viz: 最多保存多少个可视化样本
        save_dir: 保存目录

    返回:
        dict: 包含各项评估指标的字典
    """
    model.eval()

    # 指标列表
    sr_mse_list = []
    sr_psnr_list = []
    sr_ssim_list = []

    pred_mse_list = []
    pred_psnr_list = []
    pred_ssim_list = []

    gsl_err_pix_list = []
    gsl_err_m_list = []

    # 可视化数据
    viz_samples = []

    total_seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # 获取数据
            if 'lr_seq' in batch:
                # 时序模式
                inp = batch['lr_seq'].to(
                    device, non_blocking=True)  # (B, K, 1, H, W)
                # (B, 1, H_hr, W_hr)
                hr_t = batch['hr_t'].to(device, non_blocking=True)

                # ? 关键修复：hr_tp1 也必须搬到同一 device
                hr_tp1 = batch.get('hr_tp1', None)  # (B, 1, H_hr, W_hr) 或 None
                if hr_tp1 is not None:
                    hr_tp1 = hr_tp1.to(device, non_blocking=True)

                source_pos = batch['source_pos'].to(
                    device, non_blocking=True)  # (B, 2)
            else:
                # 单帧模式（兼容）
                inp = batch['lr'].to(
                    device, non_blocking=True)      # (B, 1, H, W)
                # (B, 1, H_hr, W_hr)
                hr_t = batch['hr'].to(device, non_blocking=True)
                hr_tp1 = None
                source_pos = batch['source_pos'].to(
                    device, non_blocking=True)  # (B, 2)

            # 模型推理
            sr_out, gsl_out, pred_out = model(inp)

            # 计算 SR@t 指标
            for i in range(sr_out.size(0)):
                # MSE
                mse = F.mse_loss(sr_out[i], hr_t[i])
                sr_mse_list.append(mse.item())

                # PSNR
                psnr = calculate_psnr(sr_out[i], hr_t[i], data_range=1.0)
                sr_psnr_list.append(psnr)

                # SSIM
                ssim = calculate_ssim(sr_out[i], hr_t[i], data_range=1.0)
                sr_ssim_list.append(ssim)

            # 计算 Pred@t+1 指标（如果启用且存在 hr_tp1）
            if enable_pred and (hr_tp1 is not None):
                for i in range(pred_out.size(0)):
                    # MSE
                    mse = F.mse_loss(pred_out[i], hr_tp1[i])
                    pred_mse_list.append(mse.item())

                    # PSNR
                    psnr = calculate_psnr(
                        pred_out[i], hr_tp1[i], data_range=1.0)
                    pred_psnr_list.append(psnr)

                    # SSIM
                    ssim = calculate_ssim(
                        pred_out[i], hr_tp1[i], data_range=1.0)
                    pred_ssim_list.append(ssim)

            # 计算 GSL 指标
            # 反归一化坐标（乘以 95）
            true_pos = source_pos * 95.0  # (B, 2)  (x, y) 像素坐标
            pred_pos = gsl_out * 95.0     # (B, 2)  (x, y) 像素坐标

            # 计算像素距离
            dist_pix = torch.sqrt(
                torch.sum((pred_pos - true_pos) ** 2, dim=1))  # (B,)
            gsl_err_pix_list.extend(dist_pix.detach().cpu().numpy())

            # 转换为米（除以 10）
            dist_m = dist_pix / 10.0
            gsl_err_m_list.extend(dist_m.detach().cpu().numpy())

            # 保存可视化样本（使用真实 test_set 索引命名）
            if batch_idx < max_viz:
                for i in range(min(sr_out.size(0), max_viz - len(viz_samples))):
                    if len(viz_samples) >= max_viz:
                        break

                    real_idx = total_seen + i
                    sample = {
                        'real_idx': real_idx,
                        # 最后一帧 LR
                        'lr_last': inp[i, -1].detach().cpu() if inp.dim() == 5 else inp[i].detach().cpu(),
                        'hr_t': hr_t[i].detach().cpu(),
                        'sr_t': sr_out[i].detach().cpu(),
                        # (x, y)
                        'gsl_true': true_pos[i].detach().cpu().numpy(),
                        # (x, y)
                        'gsl_pred': pred_pos[i].detach().cpu().numpy(),
                    }

                    if enable_pred and (hr_tp1 is not None):
                        sample['hr_tp1'] = hr_tp1[i].detach().cpu()
                        sample['pred_tp1'] = pred_out[i].detach().cpu()

                    viz_samples.append(sample)

            total_seen += sr_out.size(0)

    # 计算平均值（转换为 Python float 以避免 JSON 序列化问题）
    metrics = {
        'SR_MSE_mean': float(np.mean(sr_mse_list)) if len(sr_mse_list) > 0 else None,
        'SR_PSNR_mean': float(np.mean(sr_psnr_list)) if len(sr_psnr_list) > 0 else None,
        'SR_SSIM_mean': float(np.mean(sr_ssim_list)) if len(sr_ssim_list) > 0 else None,
        'GSL_err_pix_mean': float(np.mean(gsl_err_pix_list)) if len(gsl_err_pix_list) > 0 else None,
        'GSL_err_m_mean': float(np.mean(gsl_err_m_list)) if len(gsl_err_m_list) > 0 else None,
    }

    if enable_pred and len(pred_mse_list) > 0:
        metrics['Pred_MSE_mean'] = float(np.mean(pred_mse_list))
        metrics['Pred_PSNR_mean'] = float(np.mean(pred_psnr_list))
        metrics['Pred_SSIM_mean'] = float(np.mean(pred_ssim_list))

    # 保存可视化（以 sample_{真实idx} 命名）
    if save_dir and len(viz_samples) > 0:
        viz_dir = os.path.join(save_dir, 'viz')
        os.makedirs(viz_dir, exist_ok=True)

        for sample in viz_samples:
            real_idx = sample.get('real_idx', 0)
            save_visualization(sample, real_idx, viz_dir, enable_pred)

    return metrics


def save_visualization(sample, idx, save_dir, enable_pred):
    """
    保存可视化图像

    参数:
        sample: 样本数据字典
        idx: 样本索引
        save_dir: 保存目录
        enable_pred: 是否包含 Pred@t+1 可视化
    """
    # 转换为 numpy 并去除通道维度（如果是单通道）
    def to_numpy(tensor):
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            return tensor.squeeze(0).numpy()
        elif tensor.dim() == 2:
            return tensor.numpy()
        else:
            return tensor.numpy()

    lr_last = to_numpy(sample['lr_last'])
    hr_t = to_numpy(sample['hr_t'])
    sr_t = to_numpy(sample['sr_t'])

    # 计算差异图
    diff_t = np.abs(hr_t - sr_t)

    # 确定子图数量
    n_cols = 4
    n_rows = 2 if enable_pred and 'hr_tp1' in sample else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # 第一行：SR@t
    ax = axes[0, 0]
    ax.imshow(lr_last, cmap='viridis', vmin=0, vmax=1, origin='upper')
    ax.set_title('LR (last frame)')
    ax.axis('off')

    ax = axes[0, 1]
    im = ax.imshow(hr_t, cmap='viridis', vmin=0, vmax=1, origin='upper')
    ax.set_title('HR_t (Ground Truth)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    im = ax.imshow(sr_t, cmap='viridis', vmin=0, vmax=1, origin='upper')
    ax.set_title('SR_t (Predicted)')

    # 标注泄漏源位置：
    # 约定：sample['gsl_true']/['gsl_pred'] 为 (x, y) 像素坐标，其中 x=col, y=row
    # matplotlib 的 plot(x, y) 也是 (x=col, y=row)，因此这里【不需要交换】。
    true_pos = sample['gsl_true']  # (2,) -> (x, y)
    pred_pos = sample['gsl_pred']  # (2,) -> (x, y)

    ax.plot(true_pos[0], true_pos[1], 'r*', markersize=15, label='True Source')
    ax.plot(pred_pos[0], pred_pos[1], 'g*', markersize=15, label='Pred Source')

    # 可选：标注数值，方便快速核对
    ax.text(true_pos[0] + 1, true_pos[1] + 1,
            f"T({true_pos[0]:.1f},{true_pos[1]:.1f})", color='r', fontsize=7)
    ax.text(pred_pos[0] + 1, pred_pos[1] + 1,
            f"P({pred_pos[0]:.1f},{pred_pos[1]:.1f})", color='g', fontsize=7)

    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 3]
    im = ax.imshow(diff_t, cmap='hot', vmin=0, vmax=0.5, origin='upper')
    ax.set_title('Diff (HR_t - SR_t)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 第二行：Pred@t+1（如果启用）
    if enable_pred and 'hr_tp1' in sample:
        hr_tp1 = to_numpy(sample['hr_tp1'])
        pred_tp1 = to_numpy(sample['pred_tp1'])
        diff_tp1 = np.abs(hr_tp1 - pred_tp1)

        ax = axes[1, 0]
        ax.axis('off')  # 留空

        ax = axes[1, 1]
        im = ax.imshow(hr_tp1, cmap='viridis', vmin=0, vmax=1, origin='upper')
        ax.set_title('HR_{t+1} (Ground Truth)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, 2]
        im = ax.imshow(pred_tp1, cmap='viridis',
                       vmin=0, vmax=1, origin='upper')
        ax.set_title('Pred_{t+1} (Predicted)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, 3]
        im = ax.imshow(diff_tp1, cmap='hot', vmin=0, vmax=0.5, origin='upper')
        ax.set_title('Diff (HR_{t+1} - Pred_{t+1})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, f'sample_{idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def infer_single_samples(model, dataset, indices_to_evaluate, device, save_dir,
                         enable_pred, display_indices=None):
    """
    按 index 逐个推理指定样本，保存为 sample_{display_idx}

    参数:
        display_indices: 命名用索引列表，与 indices_to_evaluate 一一对应。
            若为 None 则用 indices_to_evaluate 自身命名（与单时刻对齐时传单时刻索引）
    """
    model.eval()
    sr_mse_list, sr_psnr_list, sr_ssim_list = [], [], []
    pred_mse_list, pred_psnr_list, pred_ssim_list = [], [], []
    gsl_err_pix_list, gsl_err_m_list = [], []

    viz_dir = os.path.join(save_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    if display_indices is None:
        display_indices = indices_to_evaluate

    with torch.no_grad():
        for i, real_idx in enumerate(tqdm(indices_to_evaluate, desc="Single-sample inference")):
            batch = dataset[real_idx]

            if 'lr_seq' in batch:
                inp = batch['lr_seq'].unsqueeze(
                    0).to(device, non_blocking=True)
                hr_t = batch['hr_t'].unsqueeze(0).to(device, non_blocking=True)
                hr_tp1 = batch.get('hr_tp1')
                if hr_tp1 is not None:
                    hr_tp1 = hr_tp1.unsqueeze(0).to(device, non_blocking=True)
                source_pos = batch['source_pos'].unsqueeze(0).to(
                    device, non_blocking=True)
            else:
                inp = batch['lr'].unsqueeze(0).to(device, non_blocking=True)
                hr_t = batch['hr'].unsqueeze(0).to(device, non_blocking=True)
                hr_tp1 = None
                source_pos = batch['source_pos'].unsqueeze(0).to(
                    device, non_blocking=True)

            sr_out, gsl_out, pred_out = model(inp)

            # 计算 SR@t
            mse = F.mse_loss(sr_out[0], hr_t[0])
            sr_mse_list.append(mse.item())
            sr_psnr_list.append(
                calculate_psnr(sr_out[0], hr_t[0], data_range=1.0))
            sr_ssim_list.append(
                calculate_ssim(sr_out[0], hr_t[0], data_range=1.0))

            if enable_pred and (hr_tp1 is not None):
                pred_mse_list.append(F.mse_loss(pred_out[0], hr_tp1[0]).item())
                pred_psnr_list.append(
                    calculate_psnr(pred_out[0], hr_tp1[0], data_range=1.0))
                pred_ssim_list.append(
                    calculate_ssim(pred_out[0], hr_tp1[0], data_range=1.0))

            true_pos = source_pos * 95.0
            pred_pos = gsl_out * 95.0
            dist_pix = torch.sqrt(
                torch.sum((pred_pos - true_pos) ** 2, dim=1))
            gsl_err_pix_list.extend(dist_pix.detach().cpu().numpy())
            gsl_err_m_list.extend((dist_pix / 10.0).detach().cpu().numpy())

            # 保存 viz（使用 display_idx 命名，便于与单时刻输出对应）
            disp_idx = display_indices[i] if i < len(
                display_indices) else real_idx
            lr_last = inp[0, -1] if inp.dim() == 5 else inp[0]
            sample = {
                'real_idx': disp_idx,
                'lr_last': lr_last.detach().cpu(),
                'hr_t': hr_t[0].detach().cpu(),
                'sr_t': sr_out[0].detach().cpu(),
                'gsl_true': true_pos[0].detach().cpu().numpy(),
                'gsl_pred': pred_pos[0].detach().cpu().numpy(),
            }
            if enable_pred and (hr_tp1 is not None):
                sample['hr_tp1'] = hr_tp1[0].detach().cpu()
                sample['pred_tp1'] = pred_out[0].detach().cpu()
            save_visualization(sample, disp_idx, viz_dir, enable_pred)

    metrics = {
        'SR_MSE_mean': float(np.mean(sr_mse_list)) if sr_mse_list else None,
        'SR_PSNR_mean': float(np.mean(sr_psnr_list)) if sr_psnr_list else None,
        'SR_SSIM_mean': float(np.mean(sr_ssim_list)) if sr_ssim_list else None,
        'GSL_err_pix_mean': float(np.mean(gsl_err_pix_list)) if gsl_err_pix_list else None,
        'GSL_err_m_mean': float(np.mean(gsl_err_m_list)) if gsl_err_m_list else None,
    }
    if enable_pred and pred_mse_list:
        metrics['Pred_MSE_mean'] = float(np.mean(pred_mse_list))
        metrics['Pred_PSNR_mean'] = float(np.mean(pred_psnr_list))
        metrics['Pred_SSIM_mean'] = float(np.mean(pred_ssim_list))
    return metrics


def get_test_set_indices(test_indices_str, dataset):
    """
    根据测试集索引字符串获取要评估的样本索引

    参数:
        test_indices_str: 逗号分隔的索引字符串，例如："1,2,3,4,5"
        dataset: 数据集对象

    返回:
        list: 要评估的样本索引列表
    """
    if not test_indices_str:
        return []

    try:
        indices = [int(idx.strip()) for idx in test_indices_str.split(',')]
        valid_indices = [idx for idx in indices if 0 <= idx < len(dataset)]
        if len(valid_indices) != len(indices):
            print(f"警告：部分索引超出范围，已跳过无效索引")
            print(f"可用索引范围：0 到 {len(dataset) - 1}")
            print(
                f"无效的索引：{[idx for idx in indices if idx < 0 or idx >= len(dataset)]}")
        return valid_indices
    except ValueError as e:
        print(f"错误：无效的索引格式 - {e}")
        print(f"可用索引范围：0 到 {len(dataset) - 1}")
        return []


def map_single_frame_indices_to_seq(single_test_set, seq_test_set, single_indices, K):
    """
    将单时刻 test_set 的索引映射到时序 test_set 的索引，实现样本对应。

    单时刻样本 (wind, source, time_step) 与时序样本 (wind, source, t) 对应
    当且仅当 time_step == t 且 K <= time_step <= T-1（时序需要下一帧 HR_{t+1}）。

    返回:
        list of (single_idx, seq_idx): 可映射的 (单时刻索引, 时序索引) 对
        single_idx 用于命名（与 evaluate_multi 输出一致），seq_idx 用于从 seq_test_set 取数
    """
    # 构建 (wind, source, t) -> seq_test_set 逻辑索引 的映射
    spec_to_seq_idx = {}
    for seq_idx in range(len(seq_test_set)):
        di = seq_test_set.data_indices[seq_test_set.index_list[seq_idx]]
        key = (di['wind_group'], di['source_group'], di['t'])
        spec_to_seq_idx[key] = seq_idx

    result = []
    for single_idx in single_indices:
        if single_idx < 0 or single_idx >= len(single_test_set):
            continue
        di = single_test_set.data_indices[
            single_test_set.index_list[single_idx]]
        wind = di['wind_group']
        source = di['source_group']
        time_step = di['time_step']

        if time_step < K:
            print(f"  跳过 单时刻索引 {single_idx} ({wind},{source},t={time_step}): "
                  f"时序模型需要 t>={K}")
            continue

        key = (wind, source, time_step)
        seq_idx = spec_to_seq_idx.get(key)
        if seq_idx is None:
            print(f"  跳过 单时刻索引 {single_idx} ({wind},{source},t={time_step}): "
                  f"时序无该帧（需存在 HR_{time_step+1}）")
            continue

        result.append((single_idx, seq_idx))
        print(
            f"  映射: 单时刻[{single_idx}] ({wind},{source},t={time_step}) -> 时序[{seq_idx}]")

    return result


def get_dataset_indices_seq(sample_specs, dataset):
    """
    根据样本规格获取时序数据集中的实际索引，严格按 sample_specs 顺序返回。
    规格格式与单时刻一致：wind_group,source_group,time_step（如 wind1_0,s1,50）
    时序中 time_step 对应 t（同一物理帧）
    """
    actual_indices = []
    if not sample_specs:
        return actual_indices

    index_map = {}
    for idx in range(len(dataset)):
        try:
            data_info = dataset.data_indices[dataset.index_list[idx]]
            key = f"{data_info['wind_group']},{data_info['source_group']},{data_info['t']}"
            index_map[key] = idx
        except Exception:
            continue

    for spec in sample_specs:
        try:
            idx = index_map.get(spec)
            if idx is not None:
                actual_indices.append(idx)
                print(f"找到匹配样本: {spec}, 索引={idx}")
            else:
                print(f"未找到匹配样本: {spec}")
        except Exception:
            continue

    return actual_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SwinIRMulti Sequential Model")

    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .h5 dataset file")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save evaluation results")

    # 模型参数
    parser.add_argument("--upsampler", type=str, default="nearest+conv",
                        choices=["nearest+conv", "pixelshuffle"],
                        help="Upsampler type")

    # 数据参数
    parser.add_argument("--use_seq", action="store_true",
                        help="Use sequential dataset")
    parser.add_argument("--K", type=int, default=6,
                        help="History length K (for seq dataset)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loading workers")

    # 测试模式（与 evaluate_multi.py 一致）
    parser.add_argument("--test_mode", type=str, default="all_test_set",
                        choices=["generalization", "test_set",
                                 "all_generalization", "all_test_set"],
                        help="测试模式：generalization/test_set/all_generalization/all_test_set")

    # 样本选择参数
    parser.add_argument("--sample_specs", type=str, default=None,
                        help="泛化测试的样本规格，用分号分隔，例如：wind1_0,s1,50;wind2_0,s2,30")
    parser.add_argument("--test_indices", type=str, default=None,
                        help="测试集索引，用逗号分隔，例如：1,2,3,4,5")

    # 评估参数
    parser.add_argument("--enable_pred", action="store_true",
                        help="Enable Pred@t+1 evaluation")
    parser.add_argument("--max_viz", type=int, default=10,
                        help="Maximum number of visualization samples to save")

    # 设备
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use")

    return parser.parse_args()


def main():
    args = parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # 时序评估必须使用 use_seq，强制启用
    use_seq = True

    # 根据 test_mode 构造 dataset 和 indices_to_evaluate
    dataset = None
    indices_to_evaluate = []
    display_indices = None  # 单样本推理时用于命名的索引（与单时刻对齐时=单时刻索引）
    use_single_infer = False  # 是否走单样本推理路径（不用 DataLoader）

    if args.test_mode == "generalization":
        if args.sample_specs is None:
            print("错误：generalization 模式需要提供 --sample_specs 参数")
            return
        dataset = MultiTaskSeqDataset(
            args.data_path, index_list=None, K=args.K, shuffle=False)
        sample_specs = [s.strip() for s in args.sample_specs.split(';')]
        indices_to_evaluate = get_dataset_indices_seq(sample_specs, dataset)
        # 按 spec 顺序命名，与单时刻对齐
        display_indices = list(range(len(indices_to_evaluate)))
        use_single_infer = True
        print(f"使用泛化测试模式，样本规格：{args.sample_specs}")
    elif args.test_mode == "all_generalization":
        dataset = MultiTaskSeqDataset(
            args.data_path, index_list=None, K=args.K, shuffle=False)
        indices_to_evaluate = list(range(len(dataset)))
        print("使用泛化全量模式，评估所有样本")
        use_single_infer = False
    elif args.test_mode == "all_test_set":
        _, _, test_set = generate_train_valid_test_dataset(
            args.data_path,
            train_ratio=0.8,
            valid_ratio=0.1,
            shuffle=True,
            seed=42,
            K=args.K,
            use_seq_dataset=use_seq)
        dataset = test_set
        indices_to_evaluate = list(range(len(dataset)))
        print("使用测试集全量模式，评估所有样本")
        use_single_infer = False
    else:
        # test_set 模式：test_indices 以单时刻 test_set 为基准，映射到时序
        if args.test_indices is None:
            print("错误：test_set 模式需要提供 --test_indices 参数")
            return

        # 构建单时刻和时序 test_set（同一 split，便于样本对应）
        _, _, single_test_set = generate_train_valid_test_dataset(
            args.data_path,
            train_ratio=0.8,
            valid_ratio=0.1,
            shuffle=True,
            seed=42,
            K=args.K,
            use_seq_dataset=False)
        _, _, seq_test_set = generate_train_valid_test_dataset(
            args.data_path,
            train_ratio=0.8,
            valid_ratio=0.1,
            shuffle=True,
            seed=42,
            K=args.K,
            use_seq_dataset=True)

        single_indices = get_test_set_indices(
            args.test_indices, single_test_set)
        if not single_indices:
            print("没有有效的单时刻测试索引")
            return

        print(f"使用测试集模式，单时刻测试索引：{args.test_indices}")
        print("映射到时序样本（相同 wind,source,t 对应同一物理帧）：")
        mapped = map_single_frame_indices_to_seq(
            single_test_set, seq_test_set, single_indices, args.K)
        if not mapped:
            print("无样本可映射（时序要求 K<=t<=T-1，请检查 K 或索引）")
            return

        dataset = seq_test_set
        indices_to_evaluate = [seq_idx for _, seq_idx in mapped]
        display_indices = [single_idx for single_idx, _ in mapped]
        use_single_infer = True

    if not indices_to_evaluate and use_single_infer:
        print("没有找到要评估的样本！")
        return

    n_eval = len(indices_to_evaluate) if use_single_infer else (
        len(dataset) if dataset is not None else 0)
    print(f"待评估样本数：{n_eval}")

    # 创建模型
    print("Creating model...")
    model = SwinIRMulti(
        img_size=16,
        in_chans=1,
        upscale=6,
        img_range=1.0,
        upsampler=args.upsampler,
        window_size=8,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2.0,
    )

    # 加载权重
    load_checkpoint(model, args.model_path, device)
    model = model.to(device)

    if use_single_infer:
        metrics = infer_single_samples(
            model, dataset, indices_to_evaluate, device,
            args.save_dir, args.enable_pred, display_indices=display_indices)
    else:
        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)
        print(f"Test set size: {len(dataset)}")
        print("Evaluating model...")
        metrics = evaluate_model(
            model,
            test_loader,
            device,
            enable_pred=args.enable_pred,
            max_viz=args.max_viz,
            save_dir=args.save_dir
        )

    # 保存指标
    metrics_path = os.path.join(args.save_dir, "test_metrics.json")

    def _json_default(o):
        """JSON 序列化默认处理函数，用于处理 numpy 类型"""
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return str(o)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4, default=_json_default)

    # 打印结果（仅四项核心指标）
    psnr = metrics.get('SR_PSNR_mean')
    mse = metrics.get('SR_MSE_mean')
    ssim = metrics.get('SR_SSIM_mean')
    pos_err = metrics.get('GSL_err_m_mean')
    if psnr is not None:
        print(f"Average PSNR: {psnr:.2f} dB")
    if mse is not None:
        print(f"Average MSE: {mse:.6f}")
    if ssim is not None:
        print(f"Average SSIM: {ssim:.4f}")
    if pos_err is not None:
        print(f"Average Position Error: {pos_err:.4f} m")


if __name__ == "__main__":
    main()
