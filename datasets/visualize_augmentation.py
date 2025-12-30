"""
最开始进行超分辨率所用的数据集增强程序
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_hr_lr_pair(hr_img, lr_img, title, index):
    """绘制一对HR-LR图像"""
    plt.figure(figsize=(10, 4))
    
    # 绘制HR图像
    plt.subplot(121)
    plt.imshow(hr_img, cmap='viridis')
    plt.title(f'HR - {title}')
    plt.axis('off')
    
    # 绘制LR图像
    plt.subplot(122)
    plt.imshow(lr_img, cmap='viridis')
    plt.title(f'LR - {title}')
    plt.axis('off')
    
    plt.suptitle(f'样本索引: {index}')
    plt.tight_layout()

def visualize_augmentations(original_path, augmented_path, sample_index=0):
    """可视化原始数据和增强后的数据"""
    # 读取原始数据
    with h5py.File(original_path, 'r') as f:
        original_hr = f['HR'][sample_index]
        original_lr = f['LR'][sample_index]
    
    # 读取增强后的数据
    with h5py.File(augmented_path, 'r') as f:
        aug_hr = f['HR'][sample_index::len(f['HR'])//6]  # 获取同一样本的所有增强版本
        aug_lr = f['LR'][sample_index::len(f['LR'])//6]
    
    # 绘制原始数据
    plot_hr_lr_pair(original_hr, original_lr, "原始图像", sample_index)
    
    # 绘制增强后的数据
    titles = ["旋转90度", "旋转180度", "旋转270度", "水平翻转", "垂直翻转"]
    for i, (hr, lr, title) in enumerate(zip(aug_hr[1:], aug_lr[1:], titles), 1):
        plot_hr_lr_pair(hr, lr, title, f"{sample_index} - {title}")
    
    plt.show()

def display_dataset_info(file_path):
    """显示数据集的基本信息"""
    with h5py.File(file_path, 'r') as f:
        hr_shape = f['HR'].shape
        lr_shape = f['LR'].shape
        
        print(f"数据集信息 ({file_path}):")
        print(f"HR数据形状: {hr_shape}")
        print(f"LR数据形状: {lr_shape}")
        print(f"样本数量: {hr_shape[0]}")
        print(f"HR分辨率: {hr_shape[1]}x{hr_shape[2]}")
        print(f"LR分辨率: {lr_shape[1]}x{lr_shape[2]}")
        
        # 显示数值范围
        print(f"\nHR数值范围: [{f['HR'][:].min():.4f}, {f['HR'][:].max():.4f}]")
        print(f"LR数值范围: [{f['LR'][:].min():.4f}, {f['LR'][:].max():.4f}]")

if __name__ == '__main__':
    original_path = 'C:\\Users\\yy143\\Desktop\\dataset\\old_dataset\\dataset_HRLR.h5'
    augmented_path = 'C:\\Users\\yy143\\Desktop\\dataset\\old_dataset\\dataset_HRLR_augmented.h5'
    
    # 显示数据集信息
    print("原始数据集:")
    display_dataset_info(original_path)
    print("\n增强后数据集:")
    display_dataset_info(augmented_path)
    
    # 可视化指定索引的样本及其增强版本
    sample_index = 96  # 可以修改这个值来查看不同的样本
    print(f"\n显示样本 {sample_index} 的所有增强版本...")
    visualize_augmentations(original_path, augmented_path, sample_index) 