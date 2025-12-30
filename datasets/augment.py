import h5py
import numpy as np

def rotate_90(data):
    """将数据顺时针旋转90度"""
    return np.rot90(data, k=1, axes=(1, 2))

def rotate_180(data):
    """将数据顺时针旋转180度"""
    return np.rot90(data, k=2, axes=(1, 2))

def rotate_270(data):
    """将数据顺时针旋转270度"""
    return np.rot90(data, k=3, axes=(1, 2))

def flip_horizontal(data):
    """水平翻转"""
    return np.flip(data, axis=2)

def flip_vertical(data):
    """垂直翻转"""
    return np.flip(data, axis=1)

def augment_dataset(input_path, output_path):
    """对数据集进行数据增强"""
    with h5py.File(input_path, 'r') as f:
        hr_data = f['HR'][:]
        lr_data = f['LR'][:]
        
    # 原始数据
    hr_augmented = [hr_data]
    lr_augmented = [lr_data]
    
    # 旋转增强
    hr_augmented.extend([
        rotate_90(hr_data),
        rotate_180(hr_data),
        rotate_270(hr_data)
    ])
    lr_augmented.extend([
        rotate_90(lr_data),
        rotate_180(lr_data),
        rotate_270(lr_data)
    ])
    
    # 翻转增强
    hr_augmented.extend([
        flip_horizontal(hr_data),
        flip_vertical(hr_data)
    ])
    lr_augmented.extend([
        flip_horizontal(lr_data),
        flip_vertical(lr_data)
    ])
    
    # 合并所有增强后的数据
    hr_final = np.concatenate(hr_augmented, axis=0)
    lr_final = np.concatenate(lr_augmented, axis=0)
    
    # 保存增强后的数据集
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('HR', data=hr_final, dtype=np.float32)
        f.create_dataset('LR', data=lr_final, dtype=np.float32)
    
    print(f"数据增强完成，保存至：{output_path}")
    print(f"原始数据形状: HR={hr_data.shape}, LR={lr_data.shape}")
    print(f"增强后数据形状: HR={hr_final.shape}, LR={lr_final.shape}")
    print(f"增强后样本数量增加了{len(hr_augmented)}倍")

if __name__ == '__main__':
    input_path = 'C:\\Users\\yy143\\Desktop\\dataset\\dataset_HRLR.h5'
    output_path = 'C:\\Users\\yy143\\Desktop\\dataset\\dataset_HRLR_augmented.h5'
    
    augment_dataset(input_path, output_path) 