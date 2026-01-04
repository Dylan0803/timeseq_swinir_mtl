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
from datasets.h5_dataset import generate_train_valid_test_dataset


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

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # 获取数据
            if 'lr_seq' in batch:
                # 时序模式
                inp = batch['lr_seq'].to(device)  # (B, K, 1, H, W)
                hr_t = batch['hr_t'].to(device)  # (B, 1, H_hr, W_hr)
                hr_tp1 = batch.get('hr_tp1', None)  # (B, 1, H_hr, W_hr) 或 None
                source_pos = batch['source_pos'].to(device)  # (B, 2)
            else:
                # 单帧模式（兼容）
                inp = batch['lr'].to(device)  # (B, 1, H, W)
                hr_t = batch['hr'].to(device)  # (B, 1, H_hr, W_hr)
                hr_tp1 = None
                source_pos = batch['source_pos'].to(device)  # (B, 2)

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
            if enable_pred and hr_tp1 is not None:
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
            true_pos = source_pos * 95.0  # (B, 2)
            pred_pos = gsl_out * 95.0  # (B, 2)

            # 计算像素距离
            dist_pix = torch.sqrt(
                torch.sum((pred_pos - true_pos) ** 2, dim=1))  # (B,)
            gsl_err_pix_list.extend(dist_pix.cpu().numpy())

            # 转换为米（除以 10）
            dist_m = dist_pix / 10.0
            gsl_err_m_list.extend(dist_m.cpu().numpy())

            # 保存可视化样本
            if batch_idx < max_viz:
                for i in range(min(sr_out.size(0), max_viz - len(viz_samples))):
                    if len(viz_samples) >= max_viz:
                        break

                    sample = {
                        # 最后一帧 LR
                        'lr_last': inp[i, -1].cpu() if inp.dim() == 5 else inp[i].cpu(),
                        'hr_t': hr_t[i].cpu(),
                        'sr_t': sr_out[i].cpu(),
                        'gsl_true': true_pos[i].cpu().numpy(),
                        'gsl_pred': pred_pos[i].cpu().numpy(),
                    }

                    if enable_pred and hr_tp1 is not None:
                        sample['hr_tp1'] = hr_tp1[i].cpu()
                        sample['pred_tp1'] = pred_out[i].cpu()

                    viz_samples.append(sample)

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

    # 保存可视化
    if save_dir and len(viz_samples) > 0:
        viz_dir = os.path.join(save_dir, 'viz')
        os.makedirs(viz_dir, exist_ok=True)

        for idx, sample in enumerate(viz_samples):
            save_visualization(sample, idx, viz_dir, enable_pred)

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
    n_cols = 4 if enable_pred and 'hr_tp1' in sample else 4
    n_rows = 2 if enable_pred and 'hr_tp1' in sample else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # 第一行：SR@t
    ax = axes[0, 0]
    ax.imshow(lr_last, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('LR (last frame)')
    ax.axis('off')

    ax = axes[0, 1]
    im = ax.imshow(hr_t, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('HR_t (Ground Truth)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    im = ax.imshow(sr_t, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('SR_t (Predicted)')
    # 标注泄漏源位置（坐标已经是像素坐标，直接使用）
    true_pos = sample['gsl_true']  # (2,) 已经是像素坐标
    pred_pos = sample['gsl_pred']  # (2,) 已经是像素坐标
    # 注意：imshow 的坐标系统是 (y, x)，所以需要交换
    ax.plot(true_pos[1], true_pos[0], 'r*', markersize=15, label='True Source')
    ax.plot(pred_pos[1], pred_pos[0], 'g*', markersize=15, label='Pred Source')
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 3]
    im = ax.imshow(diff_t, cmap='hot', vmin=0, vmax=0.5)
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
        im = ax.imshow(hr_tp1, cmap='viridis', vmin=0, vmax=1)
        ax.set_title('HR_{t+1} (Ground Truth)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, 2]
        im = ax.imshow(pred_tp1, cmap='viridis', vmin=0, vmax=1)
        ax.set_title('Pred_{t+1} (Predicted)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, 3]
        im = ax.imshow(diff_tp1, cmap='hot', vmin=0, vmax=0.5)
        ax.set_title('Diff (HR_{t+1} - Pred_{t+1})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, f'sample_{idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()


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

    # 加载数据集
    print("Loading test dataset...")
    _, _, test_set = generate_train_valid_test_dataset(
        args.data_path,
        train_ratio=0.8,
        valid_ratio=0.1,
        shuffle=True,
        seed=42,
        K=args.K,
        use_seq_dataset=args.use_seq
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Test set size: {len(test_set)}")

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

    # 评估
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

    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for key, value in metrics.items():
        if value is not None:
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: None")
    print("="*60)
    print(f"\nResults saved to: {metrics_path}")
    if args.max_viz > 0:
        print(f"Visualizations saved to: {os.path.join(args.save_dir, 'viz')}")


if __name__ == "__main__":
    main()
