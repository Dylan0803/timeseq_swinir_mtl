import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import shutil

# 使用自定义数据集类
from datasets.h5_dataset import ConcDatasetTorch, generate_train_valid_dataset
from models.network_swinir import SwinIR
from utils import plot_loss_lines, save_args



# 单轮训练过程
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()  # 切换为训练模式
    epoch_loss = 0.0
    tbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")  # 显示进度条

    for lr, hr in tbar:
        try:
            # 确保数据维度正确
            if len(lr.shape) == 3:
                lr = lr.unsqueeze(0)  # 添加batch维度
            if len(hr.shape) == 3:
                hr = hr.unsqueeze(0)  # 添加batch维度

            lr, hr = lr.to(device), hr.to(device)  # 将数据移到 GPU/CPU
            
            optimizer.zero_grad()  # 清空梯度
            sr = model(lr)         # 前向传播得到预测图像
            loss = criterion(sr, hr)  # 计算 MSE 损失
            
            # 检查loss是否为NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected! Skipping batch...")
                continue
                
            loss.backward()        # 反向传播
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()       # 更新参数

            batch_loss = loss.item()
            epoch_loss += batch_loss
            tbar.set_postfix(loss=f"{batch_loss:.4f}")  # 实时显示损失

            # 定期清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDNN_STATUS_" in str(e):
                print(f"Warning: GPU error detected! Clearing cache and skipping batch...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

    return epoch_loss / len(dataloader)  # 返回平均训练损失


# 验证过程，不做反向传播，关闭 Dropout 等训练专用操作
@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion, device):
    model.eval()  # 切换为评估模式
    epoch_loss = 0.0

    for lr, hr in dataloader:
        # 确保数据维度正确
        if len(lr.shape) == 3:
            lr = lr.unsqueeze(0)  # 添加batch维度
        if len(hr.shape) == 3:
            hr = hr.unsqueeze(0)  # 添加batch维度

        # 确保正确地将两个张量都移到同一设备上
        lr = lr.to(device)
        hr = hr.to(device)
        
        sr = model(lr)
        loss = criterion(sr, hr)
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)  # 返回平均验证损失


# 根据上采样器类型配置模型参数
def get_model_config(args):
    # 基础配置
    config = {
        'upscale': args.scale,
        'in_chans': 1,
        'img_size': args.patch_size,
        'window_size': 4 if args.patch_size <= 16 else 8,
        'img_range': 1.,
        'resi_connection': '1conv',
        'upsampler': args.upsampler
    }
    
    # 根据上采样器类型进行特定配置
    if args.upsampler == 'pixelshuffle':
        # pixelshuffle模式 - 使用更强大的架构
        config.update({
            'depths': [8, 8, 8, 8],
            'embed_dim': 96,
            'num_heads': [8, 8, 8, 8],
            'mlp_ratio': 4
        })
    elif args.upsampler == 'pixelshuffledirect':
        # pixelshuffledirect模式 - 轻量级架构
        config.update({
            'depths': [6, 6, 6, 6],
            'embed_dim': 60,
            'num_heads': [6, 6, 6, 6],
            'mlp_ratio': 2
        })
    elif args.upsampler == 'nearest+conv':
        # nearest+conv模式 - 中等架构，较好的零值区域处理
        config.update({
            'depths': [6, 6, 6, 6],
            'embed_dim': 80,
            'num_heads': [8, 8, 8, 8],
            'mlp_ratio': 3
        })
    else:
        # 默认配置
        config.update({
            'depths': [6, 6, 6, 6],
            'embed_dim': 60,
            'num_heads': [6, 6, 6, 6],
            'mlp_ratio': 2
        })
    
    return config


# 主训练流程
def train(args):
    logging.basicConfig(level=logging.INFO)
    print(f"Starting training with arguments: {args}")
    print(f"Using standard MSE loss")
    print(f"Upsampler mode: {args.upsampler}")

    # CUDA设置和内存管理
    torch.cuda.empty_cache()  # 清空GPU缓存
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    torch.backends.cudnn.deterministic = True  # 确保结果可复现
    
    # 设置设备为 GPU（如可用），否则为 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    print(f"Using device: {device}")

    # 使用 generate_train_valid_dataset 函数来划分训练集和验证集
    train_set, valid_set = generate_train_valid_dataset(
        args.data_path, train_ratio=0.8, shuffle=True)

    # 封装为 DataLoader
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)  # 减少num_workers，启用pin_memory
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)  # 减少num_workers，启用pin_memory

    # 打印数据集信息
    sample_lr, sample_hr = next(iter(train_loader))
    print(f"Data shapes - LR: {sample_lr.shape}, HR: {sample_hr.shape}")
    print(f"Data range - LR: [{sample_lr.min():.4f}, {sample_lr.max():.4f}]")
    print(f"Data range - HR: [{sample_hr.min():.4f}, {sample_hr.max():.4f}]")

    # 获取模型配置
    model_config = get_model_config(args)
    print(f"Model configuration: {model_config}")
    
    # 初始化 SwinIR 模型
    model = SwinIR(**model_config).to(device)

    print(f"Model created with upscale factor: {args.scale}")
    print(f"Using window_size={model_config['window_size']} for {args.patch_size}x{args.patch_size} input images")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用标准MSE损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建新的保存路径结构
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # 当前时间戳
    experiment_dir = os.path.join("experiments", f"{args.exp_name}_{timestamp}")
    model_dir = os.path.join(experiment_dir, args.model_name)
    
    # 创建保存目录结构
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"保存路径: {model_dir}")

    # 保存训练参数
    save_args(args, model_dir)

    # 初始化最佳验证损失和训练历史
    best_loss = float('inf')
    train_losses = []
    valid_losses = []
    start_epoch = 0

    # 检查是否有checkpoint可以恢复
    checkpoint_path = os.path.join(model_dir, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        best_loss = checkpoint['best_loss']
        print(f"Resuming from epoch {start_epoch}")

    # 正式开始训练循环
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        valid_loss = valid_one_epoch(model, valid_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(
            f"[{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        #  每个 epoch 结束后，打印当前时间
        print(
            f"Epoch {epoch+1} finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 保存最佳模型（验证损失最低）
        if valid_loss < best_loss:
            best_loss = valid_loss
            model_path = os.path.join(model_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f" Best model saved at epoch {epoch+1} with valid loss {best_loss:.4f}")
        
        # 保存checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'best_loss': best_loss,
            'args': args,
            'model_config': model_config  # 保存模型配置
        }
        torch.save(checkpoint, os.path.join(model_dir, "latest_checkpoint.pth"))
        
        # 每10个epoch保存一次历史checkpoint
        if (epoch + 1) % 10 == 0:
            history_checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, history_checkpoint_path)
            print(f" Checkpoint saved at epoch {epoch+1}")

        # 保存训练曲线
        plot_loss_lines(args, train_losses, valid_losses)
        
        # 保存训练历史到CSV
        history_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'valid_loss': valid_losses
        })
        history_df.to_csv(os.path.join(model_dir, 'training_history.csv'), index=False)
    
    # 训练完成后，同时在顶级实验目录保存一份最佳模型的副本，便于查找
    best_model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, os.path.join(experiment_dir, f"{args.model_name}_best_model.pth"))
        print(f"最佳模型副本已保存至: {os.path.join(experiment_dir, f'{args.model_name}_best_model.pth')}")

    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Model and checkpoints saved in {model_dir}")
    print(f"Using standard MSE loss")
    print(f"Upsampler mode: {args.upsampler}")
    
    return model, train_losses, valid_losses, best_loss


# 启动训练的命令行入口
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='模型名称')
    parser.add_argument('--exp_name', type=str,
                        default='default_experiment', help='实验名称')
    parser.add_argument('--data_path', type=str, required=True, help='h5 数据路径')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size, reduce if GPU OOM')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, reduced to improve stability')
    parser.add_argument('--scale', type=int, default=6, help='放大倍数')
    parser.add_argument('--patch_size', type=int, default=16, help='LR图像的patch大小')
    parser.add_argument('--upsampler', type=str, default='pixelshuffle', 
                        choices=['pixelshuffle', 'pixelshuffledirect', 'nearest+conv'], 
                        help='上采样方法: pixelshuffle, pixelshuffledirect, nearest+conv')
    args = parser.parse_args()

    # 这个输出目录设置不再使用，但保留兼容性
    args.output_dir = os.path.join(
        './experiments', args.model_name, args.exp_name)

    train(args)  # 开始训练
