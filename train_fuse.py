import os
import sys
import argparse
import datetime
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import pandas as pd
import time
import shutil

# 确保可以从 models 和 datasets 目录导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 确保安装: pip install warmup-scheduler
from warmup_scheduler import GradualWarmupScheduler

# 导入您的模型和数据集处理函数
from models.network_swinir_multi import SwinIRMulti
from models.network_swinir_multi_enhanced import SwinIRMultiEnhanced
from models.network_swinir_hybrid import SwinIRHybrid
from models.network_swinir_fuse import SwinIRFuse
from datasets.h5_dataset import MultiTaskDataset, generate_train_valid_test_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Advanced Multi-task Model')

    # 模型选择参数
    parser.add_argument('--model_type', type=str, default='fuse',
                        choices=['original', 'enhanced', 'hybrid', 'fuse'],
                        help='选择模型类型: original, enhanced, hybrid, or fuse')

    # 数据参数
    parser.add_argument('--data_path', type=str, required=True, help='path to the H5 dataset')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save results')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')

    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay for AdamW optimizer')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

    # 数据集划分参数
    parser.add_argument('--use_test_set', action='store_true', help='whether to use a test set split')

    # === 新增：上采样相关参数 ===
    parser.add_argument('--upsampler', type=str, default='nearest+conv', choices=['nearest+conv', 'pixelshuffle'],
                        help='上采样方式: nearest+conv 或 pixelshuffle')
    parser.add_argument('--upscale', type=int, default=6, help='上采样倍数')

    args = parser.parse_args()
    return args

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(args):
    """根据参数创建模型"""
    base_params = {
        'img_size': 16, 'in_chans': 1, 'upscale': 6, 'img_range': 1.,
        'upsampler': 'nearest+conv',
        'window_size': 8, 'mlp_ratio': 2.,
        'qkv_bias': True, 'qk_scale': None, 'drop_rate': 0., 'attn_drop_rate': 0.,
        'drop_path_rate': 0.1, 'norm_layer': nn.LayerNorm, 'ape': False,
        'patch_norm': True, 'use_checkpoint': False, 'resi_connection': '1conv'
    }
    
    model_params = {
        **base_params,
        'embed_dim': 60,
        'depths': [6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6],
    }
    
    if args.model_type == 'original':
        model = SwinIRMulti(**model_params)
    elif args.model_type == 'enhanced':
        model = SwinIRMultiEnhanced(**model_params)
    elif args.model_type == 'hybrid':
        model = SwinIRHybrid(**model_params)
    elif args.model_type == 'fuse':
        model = SwinIRFuse(**model_params)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
        
    return model

def plot_loss_lines(train_history, valid_history, save_dir):
    """绘制训练和验证的损失曲线"""
    plt.figure(figsize=(18, 12))
    
    # 总损失
    plt.subplot(2, 2, 1)
    plt.plot(train_history['total_loss'], label='Train Total Loss')
    plt.plot(valid_history['total_loss'], label='Valid Total Loss')
    plt.title('Total Loss over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    # GDM损失
    plt.subplot(2, 2, 2)
    plt.plot(train_history['gdm_loss'], label='Train GDM Loss')
    plt.plot(valid_history['gdm_loss'], label='Valid GDM Loss')
    plt.title('GDM Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    # GSL损失
    plt.subplot(2, 2, 3)
    plt.plot(train_history['gsl_loss'], label='Train GSL Loss')
    plt.plot(valid_history['gsl_loss'], label='Valid GSL Loss')
    plt.title('GSL Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    # 动态损失权重 (如果存在)
    if 'log_var_gdm' in train_history and 'log_var_gsl' in train_history:
        plt.subplot(2, 2, 4)
        plt.plot(np.exp(-np.array(train_history['log_var_gdm'])), label='GDM Weight (exp(-log_var))')
        plt.plot(np.exp(-np.array(train_history['log_var_gsl'])), label='GSL Weight (exp(-log_var))')
        plt.title('Dynamic Loss Weights over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Weight'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def save_training_history(history, save_dir):
    """保存训练历史到CSV文件"""
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index_label='epoch')

def save_args(args, save_dir):
    """保存训练参数"""
    with open(os.path.join(save_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def train_model(model, train_loader, valid_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 损失函数 - 修改为GDM使用MSE，GSL使用MAE
    gdm_criterion = nn.MSELoss()  # 改为MSE损失，适合图像超分辨率任务
    gsl_criterion = nn.L1Loss()   # 改为L1损失(MAE)，适合位置预测任务
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs - 5, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cosine)
    
    # 创建保存目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(args.save_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    save_args(args, experiment_dir)
    
    # 初始化历史记录
    train_history = {'total_loss': [], 'gdm_loss': [], 'gsl_loss': [], 'log_var_gdm': [], 'log_var_gsl': []}
    valid_history = {'total_loss': [], 'gdm_loss': [], 'gsl_loss': []}
    
    best_valid_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_train_losses = {'total': 0, 'gdm': 0, 'gsl': 0}
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]', leave=True, ncols=150)
        
        for batch in train_pbar:
            # [核心修改] 移除了 online_transform(batch) 的调用
            lr, hr, source_pos = batch['lr'].to(device), batch['hr'].to(device), batch['source_pos'].to(device)
            
            optimizer.zero_grad()
            gdm_out, gsl_out = model(lr)
            
            loss_gdm = gdm_criterion(gdm_out, hr)
            loss_gsl = gsl_criterion(gsl_out, source_pos)
            
            # 动态不确定性加权计算总损失
            log_var_gdm = model.log_var_gdm
            log_var_gsl = model.log_var_gsl
            precision_gdm = torch.exp(-log_var_gdm)
            precision_gsl = torch.exp(-log_var_gsl)
            loss = (precision_gdm * loss_gdm + log_var_gdm) + (precision_gsl * loss_gsl + log_var_gsl)
            loss = torch.mean(loss)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_losses['total'] += loss.item()
            epoch_train_losses['gdm'] += loss_gdm.item()
            epoch_train_losses['gsl'] += loss_gsl.item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'gdm_MSE': f'{loss_gdm.item():.4f}', 'gsl_MAE': f'{loss_gsl.item():.4f}'})
        
        # 记录每轮的平均损失和权重参数
        train_history['total_loss'].append(epoch_train_losses['total'] / len(train_loader))
        train_history['gdm_loss'].append(epoch_train_losses['gdm'] / len(train_loader))
        train_history['gsl_loss'].append(epoch_train_losses['gsl'] / len(train_loader))
        train_history['log_var_gdm'].append(model.log_var_gdm.item())
        train_history['log_var_gsl'].append(model.log_var_gsl.item())
        
        # 验证阶段
        model.eval()
        epoch_valid_losses = {'total': 0, 'gdm': 0, 'gsl': 0}
        
        valid_pbar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Valid]', leave=True, ncols=150)
        
        with torch.no_grad():
            for batch in valid_pbar:
                lr, hr, source_pos = batch['lr'].to(device), batch['hr'].to(device), batch['source_pos'].to(device)
                gdm_out, gsl_out = model(lr)
                
                loss_gdm = gdm_criterion(gdm_out, hr)
                loss_gsl = gsl_criterion(gsl_out, source_pos)
                
                log_var_gdm, log_var_gsl = model.log_var_gdm, model.log_var_gsl
                precision_gdm, precision_gsl = torch.exp(-log_var_gdm), torch.exp(-log_var_gsl)
                loss = torch.mean((precision_gdm * loss_gdm + log_var_gdm) + (precision_gsl * loss_gsl + log_var_gsl))
                
                epoch_valid_losses['total'] += loss.item()
                epoch_valid_losses['gdm'] += loss_gdm.item()
                epoch_valid_losses['gsl'] += loss_gsl.item()
                
                valid_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        # 记录验证损失
        valid_history['total_loss'].append(epoch_valid_losses['total'] / len(valid_loader))
        valid_history['gdm_loss'].append(epoch_valid_losses['gdm'] / len(valid_loader))
        valid_history['gsl_loss'].append(epoch_valid_losses['gsl'] / len(valid_loader))
        
        # 打印和保存
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {train_history['total_loss'][-1]:.4f}, "
              f"Valid Loss: {valid_history['total_loss'][-1]:.4f}, LR: {current_lr:.6f}")

        if valid_history['total_loss'][-1] < best_valid_loss:
            best_valid_loss = valid_history['total_loss'][-1]
            torch.save(model.state_dict(), os.path.join(experiment_dir, 'best_model.pth'))
            print(f"Best model saved at epoch {epoch+1} with validation loss {best_valid_loss:.4f}")

        # 更新学习率
        scheduler.step(epoch + 1)
        
        # 绘制并保存历史记录
        plot_loss_lines(train_history, valid_history, experiment_dir)
        save_training_history({**{'train_'+k: v for k, v in train_history.items()}, 
                               **{'valid_'+k: v for k, v in valid_history.items()}}, experiment_dir)

    print(f"Training completed! Best validation loss: {best_valid_loss:.4f}")
    print(f"Model and training data saved in {experiment_dir}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    logging.basicConfig(level=logging.INFO)
    print(f"Starting training with arguments: {args}")

    if args.use_test_set:
        train_set, valid_set, _ = generate_train_valid_test_dataset(args.data_path, train_ratio=0.8, valid_ratio=0.1, shuffle=True)
    else:
        train_set, valid_set, _ = generate_train_valid_test_dataset(args.data_path, train_ratio=0.8, valid_ratio=0.2, shuffle=True)

    # 建议使用多个 worker 加快数据加载，但在 Windows 或 Jupyter Notebook 中可能需要设为 0
    num_workers = 4 if sys.platform == 'linux' else 0
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = create_model(args)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    train_model(model, train_loader, valid_loader, args)

if __name__ == "__main__":
    main()