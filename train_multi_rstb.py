# -*- coding: utf-8 -*-
from models.network_swinir_multi_gdm import SwinIRMulti as SwinIRMultiGDM
import torch.nn.functional as F
import shutil
import time
import pandas as pd
import logging
from datasets.h5_dataset import MultiTaskDataset, generate_train_valid_test_dataset
from models.network_swinir_fuse import SwinIRFuse
from models.network_swinir_multi_enhanced import SwinIRMultiEnhanced
from models.network_swinir_multi import SwinIRMulti
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import random
import json
import datetime
import argparse
import os
import sys
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SwinIR Multi-task Model')

    # 模型选择参数
    parser.add_argument('--model_type', type=str, default='original',
                        choices=['original', 'enhanced', 'fuse', 'swinir_gdm'],
                        help='选择模型类型: original, enhanced, fuse, swinir_gdm')

    # 新增：上采样器选择参数
    parser.add_argument('--upsampler', type=str, default='nearest+conv',
                        choices=['nearest+conv', 'pixelshuffle'],
                        help='选择上采样器类型: nearest+conv 或 pixelshuffle')

    # 新增：仅对 original(SwinIRMulti) 有效的 RSTB 组数参数
    parser.add_argument('--multi_rstb', type=int, default=None,
                        help='仅对 original(SwinIRMulti) 生效：设置 RSTB 组数（如 2/4/6/8/...）')

    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to the dataset')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='directory to save results')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')

    # 训练参数
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--gdm_weight', type=float, default=1.0,
                        help='weight for GDM (super-resolution) task')
    parser.add_argument('--gsl_weight', type=float, default=0.5,
                        help='weight for GSL (source localization) task')

    # 数据集划分参数
    parser.add_argument('--use_test_set', action='store_true',
                        help='是否使用测试集划分（训练集80%，验证集10%，测试集10%）')

    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def calculate_position_error(pred_pos, gt_pos):
    return torch.sqrt(torch.sum((pred_pos - gt_pos) ** 2, dim=1))


def create_model(args):
    """根据参数创建模型"""
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
        'resi_connection': '1conv'
    }

    # 融合架构模型参数（SwinIRFuse）
    fuse_params = {
        **base_params,
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
        'resi_connection': '1conv'
    }

    # GDM 模型参数（按 network_swinir_multi_gdm.py 具体实现需求）
    gdm_params = {
        **base_params,
        'window_size': 8,
        'depths': [6, 6, 6, 6],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2.,
    }

    if args.model_type == 'original':
        model = SwinIRMulti(**original_params)
    elif args.model_type == 'enhanced':
        model = SwinIRMultiEnhanced(**enhanced_params)
    elif args.model_type == 'fuse':
        model = SwinIRFuse(**fuse_params)
    elif args.model_type == 'swinir_gdm':
        model = SwinIRMultiGDM(**gdm_params)
    else:
        raise ValueError(f"未知的模型类型: {args.model_type}")

    return model


def plot_loss_lines(args, train_losses, valid_losses, train_gdm_losses, train_gsl_losses,
                    valid_gdm_losses, valid_gsl_losses, save_dir):
    """绘制训练损失曲线"""
    plt.figure(figsize=(15, 10))

    # 总损失
    plt.subplot(221)
    plt.plot(train_losses, label='Train Total Loss')
    plt.plot(valid_losses, label='Valid Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # GDM 损失
    plt.subplot(222)
    plt.plot(train_gdm_losses, label='Train GDM Loss')
    plt.plot(valid_gdm_losses, label='Valid GDM Loss')
    plt.title('GDM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # GSL 损失
    if args.model_type != 'swinir_gdm':
        plt.subplot(223)
        plt.plot(train_gsl_losses, label='Train GSL Loss')
        plt.plot(valid_gsl_losses, label='Valid GSL Loss')
        plt.title('GSL Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()


def save_training_history(train_losses, valid_losses, train_gdm_losses, train_gsl_losses,
                          valid_gdm_losses, valid_gsl_losses, save_dir, model_type=None):
    """保存训练历史到 CSV 文件"""
    history = {
        'epoch': range(1, len(train_losses) + 1),
        'train_total_loss': train_losses,
        'valid_total_loss': valid_losses,
        'train_gdm_loss': train_gdm_losses,
        'valid_gdm_loss': valid_gdm_losses
    }
    if model_type != 'swinir_gdm':
        history['train_gsl_loss'] = train_gsl_losses
        history['valid_gsl_loss'] = valid_gsl_losses
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(
        save_dir, 'training_history.csv'), index=False)


def save_args(args, save_dir):
    """保存训练参数"""
    args_dict = vars(args)
    with open(os.path.join(save_dir, 'training_args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)


def train_model(model, train_loader, valid_loader, args):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 定义损失函数
    gdm_criterion = nn.MSELoss()  # 超分辨率重建损失
    gsl_criterion = nn.SmoothL1Loss()  # 源位置定位损失

    # 定义优化器
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    # 学习率调度
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存训练参数
    save_args(args, args.save_dir)

    # 初始化训练历史记录
    train_losses = []
    valid_losses = []
    train_gdm_losses = []
    train_gsl_losses = []
    valid_gdm_losses = []
    valid_gsl_losses = []

    # 训练循环
    best_valid_loss = float('inf')
    start_epoch = 0

    # 如存在 checkpoint 则恢复
    checkpoint_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            # 兼容旧版本 PyTorch（无 weights_only 参数）
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception:
            # 在安全模式下允许 argparse.Namespace 后再尝试
            try:
                import argparse as _argparse
                from torch.serialization import add_safe_globals
                add_safe_globals([_argparse.Namespace])
                checkpoint = torch.load(checkpoint_path, map_location=device)
            except Exception as e:
                raise e
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        train_gdm_losses = checkpoint['train_gdm_losses']
        train_gsl_losses = checkpoint['train_gsl_losses']
        valid_gdm_losses = checkpoint['valid_gdm_losses']
        valid_gsl_losses = checkpoint['valid_gsl_losses']
        best_valid_loss = checkpoint['best_valid_loss']
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_loss = 0
        train_gdm_loss = 0
        train_gsl_loss = 0  # 仅在多任务时用

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]',
                          leave=True, ncols=150)
        for batch in train_pbar:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            if args.model_type == 'swinir_gdm':
                gdm_out = model(lr)
                gdm_loss = gdm_criterion(gdm_out, hr)
                loss = gdm_loss
                train_gdm_loss += gdm_loss.item()
            else:
                source_pos = batch['source_pos'].to(device)
                gdm_out, gsl_out = model(lr)
                gdm_loss = gdm_criterion(gdm_out, hr)
                gsl_loss = gsl_criterion(gsl_out, source_pos)
                loss = args.gdm_weight * gdm_loss + args.gsl_weight * gsl_loss
                train_gdm_loss += gdm_loss.item()
                train_gsl_loss += gsl_loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # 更新进度条
            if args.model_type == 'swinir_gdm':
                train_pbar.set_postfix(
                    {'loss': f'{loss.item():.4f}', 'gdm_loss': f'{gdm_loss.item():.4f}'})
            else:
                train_pbar.set_postfix(
                    {'loss': f'{loss.item():.4f}', 'gdm_loss': f'{gdm_loss.item():.4f}', 'gsl_loss': f'{gsl_loss.item():.4f}'})
        train_loss /= len(train_loader)
        train_gdm_loss /= len(train_loader)
        if args.model_type != 'swinir_gdm':
            train_gsl_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        valid_loss = 0
        valid_gdm_loss = 0
        valid_gsl_loss = 0
        valid_pbar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Valid]',
                          leave=True, ncols=150)
        with torch.no_grad():
            for batch in valid_pbar:
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                if args.model_type == 'swinir_gdm':
                    gdm_out = model(lr)
                    gdm_loss = gdm_criterion(gdm_out, hr)
                    loss = gdm_loss
                    valid_gdm_loss += gdm_loss.item()
                else:
                    source_pos = batch['source_pos'].to(device)
                    gdm_out, gsl_out = model(lr)
                    gdm_loss = gdm_criterion(gdm_out, hr)
                    gsl_loss = gsl_criterion(gsl_out, source_pos)
                    loss = args.gdm_weight * gdm_loss + args.gsl_weight * gsl_loss
                    valid_gdm_loss += gdm_loss.item()
                    valid_gsl_loss += gsl_loss.item()
                valid_loss += loss.item()
                if args.model_type == 'swinir_gdm':
                    valid_pbar.set_postfix(
                        {'loss': f'{loss.item():.4f}', 'gdm_loss': f'{gdm_loss.item():.4f}'})
                else:
                    valid_pbar.set_postfix(
                        {'loss': f'{loss.item():.4f}', 'gdm_loss': f'{gdm_loss.item():.4f}', 'gsl_loss': f'{gsl_loss.item():.4f}'})
        valid_loss /= len(valid_loader)
        valid_gdm_loss /= len(valid_loader)
        if args.model_type != 'swinir_gdm':
            valid_gsl_loss /= len(valid_loader)

        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_gdm_losses.append(train_gdm_loss)
        valid_gdm_losses.append(valid_gdm_loss)
        if args.model_type != 'swinir_gdm':
            train_gsl_losses.append(train_gsl_loss)
            valid_gsl_losses.append(valid_gsl_loss)

        # 打印训练信息
        if args.model_type == 'swinir_gdm':
            print(f'\nEpoch {epoch+1}/{args.num_epochs} Summary:')
            print(f'Train Loss: {train_loss:.4f} (GDM: {train_gdm_loss:.4f})')
            print(f'Valid Loss: {valid_loss:.4f} (GDM: {valid_gdm_loss:.4f})')
        else:
            print(f'\nEpoch {epoch+1}/{args.num_epochs} Summary:')
            print(
                f'Train Loss: {train_loss:.4f} (GDM: {train_gdm_loss:.4f}, GSL: {train_gsl_loss:.4f})')
            print(
                f'Valid Loss: {valid_loss:.4f} (GDM: {valid_gdm_loss:.4f}, GSL: {valid_gsl_loss:.4f})')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print(
            f'Epoch {epoch+1} finished at {time.strftime("%Y-%m-%d %H:%M:%S")}')

        # 保存最优模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_path = os.path.join(
                args.save_dir, f'best_model_{args.model_type}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'args': args,
            }, save_path)
            print(f'Best model saved to {save_path}')

        # 保存 checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_gdm_losses': train_gdm_losses,
            'train_gsl_losses': train_gsl_losses,
            'valid_gdm_losses': valid_gdm_losses,
            'valid_gsl_losses': valid_gsl_losses,
            'best_valid_loss': best_valid_loss,
            'args': args
        }
        torch.save(checkpoint, os.path.join(
            args.save_dir, "latest_checkpoint.pth"))

        # 每 10 个 epoch 另存历史 checkpoint
        if (epoch + 1) % 10 == 0:
            history_checkpoint_path = os.path.join(
                args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, history_checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        # 绘制损失曲线
        plot_loss_lines(args, train_losses, valid_losses,
                        train_gdm_losses, train_gsl_losses,
                        valid_gdm_losses, valid_gsl_losses,
                        args.save_dir)

        # 保存训练历史
        save_training_history(train_losses, valid_losses,
                              train_gdm_losses, train_gsl_losses,
                              valid_gdm_losses, valid_gsl_losses,
                              args.save_dir, args.model_type)

    # 训练结束后，备份一份最佳模型的副本
    best_model_path = os.path.join(
        args.save_dir, f'best_model_{args.model_type}.pth')
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, os.path.join(
            args.save_dir, f"{args.model_type}_best_model.pth"))
        print(
            f"最佳模型副本已保存: {os.path.join(args.save_dir, f'{args.model_type}_best_model.pth')}")

    print("Training completed!")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Model and checkpoints saved in {args.save_dir}")

    return model, train_losses, valid_losses, best_valid_loss


def train(args):
    logging.basicConfig(level=logging.INFO)
    print(f"Starting training with arguments: {args}")
    print(f"Using multi-task learning with GDM and GSL")

    # CUDA 设置和内存处理
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    print(f"Using device: {device}")

    # 数据集载入和打印信息
    if args.use_test_set:
        train_set, valid_set, test_set = generate_train_valid_test_dataset(
            args.data_path, train_ratio=0.8, valid_ratio=0.1, shuffle=True)
        print(f"使用测试集划分模式：训练集80%，验证集10%，测试集10%")
    else:
        train_set, valid_set, _ = generate_train_valid_test_dataset(
            args.data_path, train_ratio=0.8, valid_ratio=0.2, shuffle=True)
        print(f"使用传统划分模式：训练集80%，验证集20%")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    # 打印数据集信息
    sample = next(iter(train_loader))
    print(f"Data shapes:")
    print(f"LR: {sample['lr'].shape}")
    print(f"HR: {sample['hr'].shape}")
    print(f"Source position: {sample['source_pos'].shape}")
    print(f"HR max position: {sample['hr_max_pos'].shape}")
    print(f"Data ranges:")
    print(f"LR: [{sample['lr'].min():.4f}, {sample['lr'].max():.4f}]")
    print(f"HR: [{sample['hr'].min():.4f}, {sample['hr'].max():.4f}]")
    print(
        f"Source position: [{sample['source_pos'].min():.4f}, {sample['source_pos'].max():.4f}]")
    print(
        f"HR max position: [{sample['hr_max_pos'].min():.4f}, {sample['hr_max_pos'].max():.4f}]")

    # 创建模型
    model = create_model(args)

    # 训练模型
    model, train_losses, valid_losses, best_valid_loss = train_model(
        model, train_loader, valid_loader, args)


def main():
    args = parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
