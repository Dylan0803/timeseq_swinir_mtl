import torch
from models.network_swinir import SwinIR
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

# === 命令行参数设置 ===
def parse_args():
    parser = argparse.ArgumentParser(description="SwinIR Inference")
    parser.add_argument('--model_path', type=str,
                        default='./experiments/my_exp_20250418-153000/best_model.pth', help='模型路径')
    parser.add_argument('--data_path', type=str,
                        default='/content/drive/MyDrive/1.h5', help='h5 文件路径')
    parser.add_argument('--sample_index', type=int,
                        default=0, help='测试第几个样本（比如0）')
    parser.add_argument('--scale', type=int, default=6, help='放大倍数，和训练时一致')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available()
                        else 'cpu', help='设备类型: cuda 或 cpu')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='保存比较数据的目录')
    parser.add_argument('--upsampler', type=str, default='pixelshuffle',
                       choices=['pixelshuffle', 'pixelshuffledirect', 'nearest+conv'],
                       help='上采样方法: pixelshuffle, pixelshuffledirect, nearest+conv')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='LR图像的patch大小')
    return parser.parse_args()

# === 加载命令行参数 ===
args = parse_args()

# === 参数设置 ===
model_path = args.model_path
data_path = args.data_path
sample_index = args.sample_index
scale = args.scale
device = torch.device(args.device)
output_dir = args.output_dir
upsampler = args.upsampler
patch_size = args.patch_size

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 尝试从checkpoint加载模型配置
try:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        print("使用checkpoint中的模型配置")
        model_config = checkpoint['model_config']
        model = SwinIR(**model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 手动配置模型
        print("使用命令行参数配置模型")
        window_size = 4 if patch_size <= 16 else 8
        
        # 根据上采样器类型配置模型参数
        if upsampler == 'pixelshuffle':
            model = SwinIR(
                upscale=scale,
                in_chans=1,
                img_size=patch_size,
                window_size=window_size,
                img_range=1.,
                depths=[8, 8, 8, 8],
                embed_dim=96,
                num_heads=[8, 8, 8, 8],
                mlp_ratio=4,
                upsampler=upsampler,
                resi_connection='1conv'
            ).to(device)
        elif upsampler == 'pixelshuffledirect':
            model = SwinIR(
                upscale=scale,
                in_chans=1,
                img_size=patch_size,
                window_size=window_size,
                img_range=1.,
                depths=[6, 6, 6, 6],
                embed_dim=60,
                num_heads=[6, 6, 6, 6],
                mlp_ratio=2,
                upsampler=upsampler,
                resi_connection='1conv'
            ).to(device)
        elif upsampler == 'nearest+conv':
            model = SwinIR(
                upscale=scale,
                in_chans=1,
                img_size=patch_size,
                window_size=window_size,
                img_range=1.,
                depths=[6, 6, 6, 6],
                embed_dim=80,
                num_heads=[8, 8, 8, 8],
                mlp_ratio=3,
                upsampler=upsampler,
                resi_connection='1conv'
            ).to(device)
        else:
            # 默认配置
            model = SwinIR(
                upscale=scale,
                in_chans=1,
                img_size=patch_size,
                window_size=window_size,
                img_range=1.,
                depths=[6, 6, 6, 6],
                embed_dim=60,
                num_heads=[6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv'
            ).to(device)
            
        model.load_state_dict(torch.load(model_path, map_location=device))
except Exception as e:
    print(f"加载模型时出错: {e}")
    print("使用默认模型配置")
    # 使用默认配置
    model = SwinIR(
        upscale=scale,
        in_chans=1,
        img_size=16,
        window_size=4,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

# 打印模型信息
print(f"模型已加载，使用上采样器: {upsampler}, 放大倍数: {scale}")

model.eval()

# === 加载一个测试样本 ===
with h5py.File(data_path, 'r') as f:
    lr_data = f['LR'][sample_index]  # shape: (b, b)
    hr_data = f['HR'][sample_index]  # shape: (a, a)

# 转为张量，添加 batch 和 channel 维度
lr_tensor = torch.from_numpy(lr_data).unsqueeze(
    0).unsqueeze(0).float().to(device)  # shape: [1,1,b,b]

# === 推理 ===
with torch.no_grad():
    sr_tensor = model(lr_tensor)  # shape: [1,1,a,a]
    sr_image = sr_tensor.squeeze().cpu().numpy()  # shape: [a, a]

# === 在这里添加后处理代码 ===
# 设置下限为0
sr_image = np.maximum(sr_image, 0)  # 小于0的值会被设置为0

# === 分析SR和HR数据 ===
# 计算差异
diff = hr_data - sr_image
mse = np.mean(np.square(diff))
psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
mae = np.mean(np.abs(diff))

# 打印数据统计信息
print("\n=== 数据分析结果 ===")
print(f"HR 形状: {hr_data.shape}, 最小值: {hr_data.min():.6f}, 最大值: {hr_data.max():.6f}, 均值: {hr_data.mean():.6f}")
print(f"SR 形状: {sr_image.shape}, 最小值: {sr_image.min():.6f}, 最大值: {sr_image.max():.6f}, 均值: {sr_image.mean():.6f}")
print(f"差异 - 最小值: {diff.min():.6f}, 最大值: {diff.max():.6f}, 均值: {diff.mean():.6f}")
print(f"MSE: {mse:.6f}, PSNR: {psnr:.2f} dB, MAE: {mae:.6f}")

# === 零值区域单独分析 ===
# 找出HR中的零值区域
zero_mask = hr_data < 1e-6
zero_pixels = np.count_nonzero(zero_mask)
total_pixels = hr_data.size

if zero_pixels > 0:
    zero_mse = np.mean(np.square(hr_data[zero_mask] - sr_image[zero_mask]))
    zero_mae = np.mean(np.abs(hr_data[zero_mask] - sr_image[zero_mask]))
    print("\n=== 零值区域分析 ===")
    print(f"零值区域像素数: {zero_pixels} ({zero_pixels/total_pixels*100:.2f}%)")
    print(f"零值区域 - MSE: {zero_mse:.6f}, MAE: {zero_mae:.6f}")
    
    # 非零区域的指标
    non_zero_mask = ~zero_mask
    non_zero_mse = np.mean(np.square(hr_data[non_zero_mask] - sr_image[non_zero_mask]))
    non_zero_psnr = 10 * np.log10(1.0 / non_zero_mse) if non_zero_mse > 0 else float('inf')
    non_zero_mae = np.mean(np.abs(hr_data[non_zero_mask] - sr_image[non_zero_mask]))
    
    print("\n=== 非零区域分析 ===")
    print(f"非零区域 - MSE: {non_zero_mse:.6f}, PSNR: {non_zero_psnr:.2f} dB, MAE: {non_zero_mae:.6f}")
    print(f"非零区域像素数: {np.count_nonzero(non_zero_mask)} ({np.count_nonzero(non_zero_mask)/total_pixels*100:.2f}%)")

# === 保存统计结果为CSV ===
stats_df = pd.DataFrame({
    '指标': ['最小值', '最大值', '均值', 'MSE', 'PSNR', 'MAE'],
    'HR': [hr_data.min(), hr_data.max(), hr_data.mean(), np.nan, np.nan, np.nan],
    'SR': [sr_image.min(), sr_image.max(), sr_image.mean(), np.nan, np.nan, np.nan],
    '差异': [diff.min(), diff.max(), diff.mean(), mse, psnr, mae]
})
stats_file = os.path.join(output_dir, f'sample_{sample_index}_stats.csv')
stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
print(f"统计数据已保存到 {stats_file}")

# === 保存图像数据为CSV ===
# 转换2D数组为1D并创建DataFrame
hr_flat = hr_data.flatten()
sr_flat = sr_image.flatten()
diff_flat = diff.flatten()

# 创建位置索引
positions = [f"({i//hr_data.shape[1]},{i%hr_data.shape[1]})" for i in range(len(hr_flat))]

# 创建数据DataFrame
data_df = pd.DataFrame({
    '位置': positions,
    'HR': hr_flat,
    'SR': sr_flat,
    '差异': diff_flat
})

# 保存为CSV
data_file = os.path.join(output_dir, f'sample_{sample_index}_data.csv')
data_df.to_csv(data_file, index=False, encoding='utf-8-sig')
print(f"像素数据已保存到 {data_file}")

# === 可视化对比并保存 ===
plt.figure(figsize=(20, 5))  # 调整图像大小以适应横向布局

# 原始图像对比
plt.subplot(1, 4, 1)
plt.title("Ground Truth HR")
plt.imshow(hr_data, cmap='viridis')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("LR input")
plt.imshow(lr_data, cmap='viridis')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title(f"SR output ({upsampler})")
plt.imshow(sr_image, cmap='viridis')
plt.colorbar()
plt.axis('off')

# 添加差异图
plt.subplot(1, 4, 4)
plt.title("Difference (HR - SR)")
diff_plot = plt.imshow(diff, cmap='coolwarm')
plt.colorbar(diff_plot)
plt.axis('off')

# 保存为图像文件，而不是直接展示
plt.tight_layout()
image_file = os.path.join(output_dir, f'sample_{sample_index}_comparison.png')
plt.savefig(image_file, dpi=300)
print(f"分析图像已保存为 {image_file}")