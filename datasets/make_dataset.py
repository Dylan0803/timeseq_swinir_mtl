"""
用来将txt浓度数据转换为h5文件，同时按照不同的条件进行划分，加上泄漏源浓度位置以及风场等信息，并针对HR进行下采样
"""
import h5py
import numpy as np
import os
from tqdm.auto import tqdm
from scipy.ndimage import zoom  # 添加这个导入

def read_wind_data(wind_file):
    """
    读取风场数据
    
    Args:
        wind_file: 风场CSV文件路径
    
    Returns:
        wind_data: 包含位置和速度矢量的字典
    """
    # 读取CSV文件
    data = np.loadtxt(wind_file, delimiter=',', skiprows=1)  # 跳过标题行
    
    # 提取数据
    points = data[:, :2]  # Points:0, Points:1
    velocity = data[:, 2:]  # U:0, U:1
    
    return {
        'points': points,
        'velocity': velocity
    }

def read_concentration_file(file_path):
    """
    读取浓度数据文件
    
    Args:
        file_path: 浓度文件路径
    
    Returns:
        metadata: 文件元数据（z值、迭代次数等）
        data: 浓度数据数组
    """
    metadata = {}
    data = []
    
    with open(file_path, 'r') as f:
        # 读取元数据
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value = line.strip('# ').split(':')
                    metadata[key.strip()] = value.strip()
            else:
                # 读取数据行
                if line.strip():  # 跳过空行
                    x, y, conc = map(float, line.strip().split())
                    data.append([x, y, conc])
    
    # 转换为numpy数组
    data = np.array(data)
    
    # 获取宽度和高度
    width = int(metadata['Width'])
    height = int(metadata['Height'])
    
    # 重塑数据为2D数组
    conc_data = np.zeros((height, width))
    for x, y, conc in data:
        conc_data[int(y), int(x)] = conc
    
    # 垂直翻转数据
    conc_data = np.flipud(conc_data)
    
    return metadata, conc_data

def get_source_positions(source_file):
    """
    获取所有泄漏源位置信息
    
    Args:
        source_file: 源位置文件路径
    
    Returns:
        source_positions: 包含所有泄漏源位置信息的字典
    """
    source_positions = {}
    
    with open(source_file, 'r') as f:
        for line in f:
            if line.strip():
                # 解析行数据
                parts = line.strip().split('-------')
                if len(parts) == 2:
                    key = parts[0].strip()
                    coords = parts[1].strip().split()
                    
                    # 提取x和y坐标
                    x = float(coords[0].split(':')[1])
                    y = float(coords[1].split(':')[1])
                    
                    # 转换坐标（乘以10并取整）
                    x = int(x * 10)
                    y = int(y * 10)

                    source_positions[key] = np.array([x, y])
    
    return source_positions

def txt_to_h5(data_root, output_path, wind_files):
    """
    将txt浓度数据转换为h5文件，并进行下采样
    
    Args:
        data_root: 原始数据根目录，包含w1s1-w1s8等文件夹
        output_path: 输出h5文件路径
        wind_files: 风场文件路径字典
    """
    # 获取泄漏源位置信息
    source_file = os.path.join(data_root, 'source.txt')
    source_positions = get_source_positions(source_file)
    
    # 创建h5文件
    with h5py.File(output_path, 'w') as f:
        # 动态获取存在的风场和源组合
        existing_combinations = []
        for wind_idx in range(1, 4):
            for source_idx in range(1, 9):
                source_dir = os.path.join(data_root, f'w{wind_idx}s{source_idx}')
                if os.path.exists(source_dir):
                    existing_combinations.append((wind_idx, source_idx))
        
        # 处理风场数据
        for wind_idx, wind_file in wind_files.items():
            # 读取风场数据
            wind_data = read_wind_data(wind_file)
            
            # 创建风场组
            wind_group = f.create_group(wind_idx)
            
            # 存储风场数据
            wind_group.create_dataset('points', data=wind_data['points'])
            wind_group.create_dataset('velocity', data=wind_data['velocity'])
        
        # 处理存在的风场和源组合
        for wind_idx, source_idx in existing_combinations:
            wind_group = f[f'wind{wind_idx}']
            source_group = wind_group.create_group(f's{source_idx}')
            
            # 构建源数据路径
            source_dir = os.path.join(data_root, f'w{wind_idx}s{source_idx}')
            
            # 获取所有concentration文件
            conc_files = sorted([f for f in os.listdir(source_dir) 
                              if f.startswith('concentration_') and f.endswith('.txt')])
            
            # 读取并存储每个浓度文件
            for i, conc_file in enumerate(tqdm(conc_files, 
                desc=f'Processing wind{wind_idx} source{source_idx}')):
                
                # 读取txt文件
                metadata, conc_data = read_concentration_file(
                    os.path.join(source_dir, conc_file)
                )
                
                # 剪切数据到96x96
                conc_data = conc_data[2:98, 2:98]  # 从中心剪切出96x96的数据
                
                # 存储高分辨率数据
                dataset = source_group.create_dataset(
                    f'HR_{i+1}', 
                    data=conc_data,
                    compression='gzip'  # 使用gzip压缩
                )
                
                # 下采样到16x16
                scale_factor = 16 / 96  # 计算缩放因子
                lr_data = zoom(conc_data, scale_factor, order=1)  # 使用线性插值
                
                # 存储低分辨率数据
                lr_dataset = source_group.create_dataset(
                    f'LR_{i+1}',
                    data=lr_data,
                    compression='gzip'
                )
                
                # 存储元数据
                for key, value in metadata.items():
                    dataset.attrs[key] = value
                    lr_dataset.attrs[key] = value
            
            # 存储源位置信息
            source_key = f'w{wind_idx}s{source_idx}'
            if source_key in source_positions:
                # 获取源位置
                source_pos = source_positions[source_key]
                
                # 调整源位置坐标
                if wind_idx in [1, 3] and source_idx in [1, 2, 3]:
                    # 对于w1s1-w1s3和w3s1-w3s3，不需要额外的偏移
                    pass
                else:
                    # 对于其他情况，确保不会出现负值
                    source_pos[0] = max(0, source_pos[0] - 2)
                    source_pos[1] = max(0, source_pos[1] - 2)
                
                # 获取数据高度
                height = source_group['HR_1'].shape[0]
                # 垂直翻转y坐标
                source_pos[1] = height - 1 - source_pos[1]
                # 创建源信息数组，包含位置和浓度信息
                source_info = np.array([source_pos[0], source_pos[1], 10.0])  # 10.0代表10ppm
                # 存储源信息
                source_group.create_dataset('source_info', data=source_info)

def augment_dataset(h5_path, output_path):
    """
    对数据集进行增强处理，并添加调试输出
    """
    print("\n开始数据增强处理...")
    with h5py.File(h5_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # 遍历所有风场组
        for wind_group_name in f_in.keys():
            if not wind_group_name.startswith('wind'):
                continue
                
            wind_idx = int(wind_group_name.replace('wind', ''))
            print(f"\n处理风场 {wind_idx}...")
            wind_group = f_in[wind_group_name]
            
            # Create original data group (with _0 suffix)
            wind_out_group = f_out.create_group(f'wind{wind_idx}_0')
            wind_out_group.create_dataset('points', data=wind_group['points'][:])
            wind_out_group.create_dataset('velocity', data=wind_group['velocity'][:])
            
            # 获取该风场下实际存在的源位置
            existing_sources = [key for key in wind_group.keys() if key.startswith('s')]
            
            # Process each existing source position
            for source_key in existing_sources:
                source_idx = int(source_key.replace('s', ''))
                print(f"  处理源位置 {source_idx}...")
                source_group = wind_group[source_key]
                source_out_group = wind_out_group.create_group(source_key)
                
                # Copy original data
                for i in range(1, len([k for k in source_group.keys() if k.startswith('HR_')]) + 1):
                    source_out_group.create_dataset(f'HR_{i}', data=source_group[f'HR_{i}'][:], compression='gzip')
                    source_out_group.create_dataset(f'LR_{i}', data=source_group[f'LR_{i}'][:], compression='gzip')
                    # Copy metadata
                    for key, value in source_group[f'HR_{i}'].attrs.items():
                        source_out_group[f'HR_{i}'].attrs[key] = value
                        source_out_group[f'LR_{i}'].attrs[key] = value
                
                # Copy source position information
                source_out_group.create_dataset('source_info', data=source_group['source_info'][:])
            
            # Define augmentation operations
            augmentations = {
                'rot90': lambda x: np.rot90(x, k=1),  # Counter-clockwise 90°
                'rot180': lambda x: np.rot90(x, k=2),  # Counter-clockwise 180°
                'rot270': lambda x: np.rot90(x, k=3),  # Counter-clockwise 270°
                'flip_v': lambda x: np.flipud(x),      # Vertical flip
                'flip_h': lambda x: np.fliplr(x)       # Horizontal flip
            }
            
            # Process each augmentation operation
            for aug_name, aug_func in augmentations.items():
                print(f"  处理增强操作: {aug_name}...")
                # Create augmented wind field group
                wind_aug_group = f_out.create_group(f'wind{wind_idx}_{aug_name}')
                
                # Process wind field data
                points = wind_group['points'][:]
                velocity = wind_group['velocity'][:]
                
                # Process wind field data based on augmentation type
                if 'rot' in aug_name:
                    # Rotation operations
                    center = np.array([48, 48])
                    relative_points = points - center
                    if aug_name == 'rot90':
                        # Counter-clockwise 90° rotation
                        rotation_matrix = np.array([[0, -1], [1, 0]])
                        velocity = np.dot(velocity, rotation_matrix.T)
                    elif aug_name == 'rot180':
                        # Counter-clockwise 180° rotation
                        rotation_matrix = np.array([[-1, 0], [0, -1]])
                        velocity = np.dot(velocity, rotation_matrix.T)
                    else:  # rot270
                        # Counter-clockwise 270° rotation
                        rotation_matrix = np.array([[0, 1], [-1, 0]])
                        velocity = np.dot(velocity, rotation_matrix.T)
                    
                    # Rotate points
                    relative_points = np.dot(relative_points, rotation_matrix.T)
                    points = relative_points + center
                else:
                    # Flip operations
                    if aug_name == 'flip_v':
                        points[:, 1] = 96 - points[:, 1]
                        velocity[:, 1] = -velocity[:, 1]
                    else:  # flip_h
                        points[:, 0] = 96 - points[:, 0]
                        velocity[:, 0] = -velocity[:, 0]
                
                wind_aug_group.create_dataset('points', data=points)
                wind_aug_group.create_dataset('velocity', data=velocity)
                
                # Process each existing source position
                for source_key in existing_sources:
                    source_group = wind_group[source_key]
                    source_aug_group = wind_aug_group.create_group(source_key)
                    
                    # Process each time step
                    for i in range(1, len([k for k in source_group.keys() if k.startswith('HR_')]) + 1):
                        # Augment concentration data
                        hr_data = aug_func(source_group[f'HR_{i}'][:])
                        lr_data = aug_func(source_group[f'LR_{i}'][:])
                        
                        source_aug_group.create_dataset(f'HR_{i}', data=hr_data, compression='gzip')
                        source_aug_group.create_dataset(f'LR_{i}', data=lr_data, compression='gzip')
                        
                        # Copy metadata
                        for key, value in source_group[f'HR_{i}'].attrs.items():
                            source_aug_group[f'HR_{i}'].attrs[key] = value
                            source_aug_group[f'LR_{i}'].attrs[key] = value
                    
                    # Process source position information
                    source_info = source_group['source_info'][:]
                    x, y, conc = source_info
                    if aug_name == 'rot90':
                        # 逆时针90度旋转：x, y → y, 95 - x
                        x_new = y
                        y_new = 95 - x
                    elif aug_name == 'rot180':
                        # 逆时针180度旋转：x, y → 95 - x, 95 - y
                        x_new = 95 - x
                        y_new = 95 - y
                    elif aug_name == 'rot270':
                        # 逆时针270度旋转：x, y → 95 - y, x
                        x_new = 95 - y
                        y_new = x
                    elif aug_name == 'flip_v':
                        # 上下翻转：y → 95 - y
                        x_new = x
                        y_new = 95 - y
                    elif aug_name == 'flip_h':
                        # 左右翻转：x → 95 - x
                        x_new = 95 - x
                        y_new = y
                    else:
                        x_new = x
                        y_new = y
                    source_aug_group.create_dataset('source_info', data=np.array([x_new, y_new, conc]))
    
    print("\n数据增强处理完成！")

def normalize_dataset(h5_path, output_path):
    """
    对增强后的数据集进行归一化处理，对每种情况的HR和LR对分别进行归一化
    归一化后的数据直接以HR和LR的形式保存
    """
    print("\n开始归一化数据...")
    with h5py.File(h5_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # 遍历所有风场组
        for wind_group_name in f_in.keys():
            print(f"\n处理风场组: {wind_group_name}...")
            wind_group = f_in[wind_group_name]
            wind_out_group = f_out.create_group(wind_group_name)
            
            # 复制风场数据（不归一化）
            wind_out_group.create_dataset('points', data=wind_group['points'][:])
            wind_out_group.create_dataset('velocity', data=wind_group['velocity'][:])
            
            # 遍历所有源位置
            for source_idx in range(1, 9):
                source_key = f's{source_idx}'
                if source_key not in wind_group:
                    continue
                
                print(f"  处理源位置 {source_idx}...")
                source_group = wind_group[source_key]
                source_out_group = wind_out_group.create_group(source_key)
                
                # 获取所有时间步
                time_steps = [k for k in source_group.keys() if k.startswith('HR_')]
                
                # 对每个时间步的HR和LR对进行归一化
                for time_step in time_steps:
                    hr_key = time_step
                    lr_key = f'LR_{time_step.split("_")[1]}'
                    
                    # 读取HR和LR数据
                    hr_data = source_group[hr_key][:]
                    lr_data = source_group[lr_key][:]
                    
                    # 计算该对数据的最大最小值
                    max_val = max(hr_data.max(), lr_data.max())
                    min_val = min(hr_data.min(), lr_data.min())
                    
                    # 归一化HR和LR数据
                    normalized_hr = (hr_data - min_val) / (max_val - min_val)
                    normalized_lr = (lr_data - min_val) / (max_val - min_val)
                    
                    # 存储归一化后的数据（直接以HR和LR的形式保存）
                    source_out_group.create_dataset(hr_key, data=normalized_hr, compression='gzip')
                    source_out_group.create_dataset(lr_key, data=normalized_lr, compression='gzip')
                    
                    # 复制元数据
                    for attr_key, attr_value in source_group[hr_key].attrs.items():
                        source_out_group[hr_key].attrs[attr_key] = attr_value
                        source_out_group[lr_key].attrs[attr_key] = attr_value
                
                # 复制源位置信息（不归一化）
                source_out_group.create_dataset('source_info', data=source_group['source_info'][:])
        
        print("\n归一化完成！")

def main():
    # 设置所有路径
    base_dir = 'C:\\Users\\yy143\\Desktop\\dataset'
    data_root = os.path.join(base_dir, 'test_data')  # 原始数据根目录
    source_dataset_dir = os.path.join(base_dir, 'source_dataset')  # 输出目录
    
    # 创建输出目录（如果不存在）
    os.makedirs(source_dataset_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_path = os.path.join(source_dataset_dir, 'test_dataset.h5')
    aug_output_path = os.path.join(source_dataset_dir, 'test_augmented_dataset.h5')
    norm_output_path = os.path.join(source_dataset_dir, 'test_normalized_augmented_dataset.h5')
    
    # 设置风场文件路径
    wind_files = {
        'wind1': os.path.join(base_dir, 'test_data', 'inlet11_05.csv'),
        'wind2': os.path.join(base_dir, 'test_data', 'inlet22_05.csv'),
        'wind3': os.path.join(base_dir, 'test_data', 'inlet33_05.csv')
    }
    
    # 转换数据
    print("\n开始转换数据...")
    txt_to_h5(data_root, output_path, wind_files)
    print("数据转换完成！")
    
    # 进行数据增强
    augment_dataset(output_path, aug_output_path)
    
    # 进行数据归一化
    normalize_dataset(aug_output_path, norm_output_path)

if __name__ == '__main__':
    main()
