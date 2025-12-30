"""
用来将数据集可视化，并查看数据集的基本信息
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

def visualize_concentration_and_wind(h5_path):
    """
    可视化h5文件中的浓度数据和风场数据
    
    Args:
        h5_path: h5文件路径
    """
    with h5py.File(h5_path, 'r') as f:
        # 获取所有风场和源位置
        wind_groups = list(f.keys())
        source_groups = [k for k in f[wind_groups[0]].keys() if k.startswith('s')]
        
        # 获取第一个数据集的形状
        first_data = f[wind_groups[0]][source_groups[0]]['HR_1'][:]
        
        # 计算实际的迭代次数
        num_iterations = len([k for k in f[wind_groups[0]][source_groups[0]].keys() 
                            if k.startswith('HR_')])  # 只计算HR_开头的键的数量
        
        # 创建图形和子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        plt.subplots_adjust(bottom=0.3)  # 为滑块留出空间
        
        # 创建滑块
        wind_ax = plt.axes([0.2, 0.2, 0.6, 0.03])
        source_ax = plt.axes([0.2, 0.15, 0.6, 0.03])
        iter_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
        
        wind_slider = Slider(wind_ax, 'Wind', 1, len(wind_groups), valinit=1, valstep=1)
        source_slider = Slider(source_ax, 'Source', 1, len(source_groups), valinit=1, valstep=1)
        iter_slider = Slider(iter_ax, 'Iteration', 1, num_iterations, valinit=1, valstep=1)
        
        # 创建颜色条
        im1 = ax1.imshow(first_data, cmap='viridis')
        cbar1 = plt.colorbar(im1, ax=ax1, label='Concentration')
        
        # 获取风场数据
        wind_points = f[wind_groups[0]]['points'][:]
        wind_velocity = f[wind_groups[0]]['velocity'][:]
        
        # 绘制风场矢量图
        im2 = ax2.quiver(wind_points[:, 0], wind_points[:, 1], 
                        wind_velocity[:, 0], wind_velocity[:, 1],
                        scale=50)
        ax2.set_title('Wind Field')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        # 初始化LR图的显示
        im3 = ax3.imshow(np.zeros((16, 16)), cmap='viridis')
        cbar3 = plt.colorbar(im3, ax=ax3, label='Concentration')
        
        def update(val):
            wind_idx = int(wind_slider.val)
            source_idx = int(source_slider.val)
            iter_idx = int(iter_slider.val)
            
            wind_group_name = f'wind{wind_idx}'
            source_group_name = f's{source_idx}'
            
            try:
                # 更新HR浓度数据
                wind_group = f[wind_group_name]
                source_group = wind_group[source_group_name]
                hr_data = source_group[f'HR_{iter_idx}'][:]
                lr_data = source_group[f'LR_{iter_idx}'][:]
                
                # 更新HR图
                im1.set_data(hr_data)
                im1.set_clim(vmin=hr_data.min(), vmax=hr_data.max())
                ax1.set_title(f'HR Concentration - Wind {wind_idx}, Source {source_idx}, Iteration {iter_idx}')
                
                # 更新LR图
                im3.set_data(lr_data)
                im3.set_clim(vmin=lr_data.min(), vmax=lr_data.max())
                ax3.set_title(f'LR Concentration (16x16)')
                
                # 清除之前的所有标记和文本
                for artist in ax1.lines + ax1.texts:
                    artist.remove()
                if ax1.legend_:
                    ax1.legend_.remove()
                
                if 'source_info' in source_group:
                    source_info = source_group['source_info'][:]
                    # 只使用前两个数据（位置信息）
                    source_pos = source_info[:2]
                    # 绘制泄漏源位置，使用红星标记
                    ax1.plot(source_pos[0], source_pos[1], 'r*', markersize=5, label='Source')
                    # 添加坐标信息，位置调整到右上角
                    ax1.text(source_pos[0]+5, source_pos[1]+5, 
                            f'({source_pos[0]/10:.1f}, {source_pos[1]/10:.1f})',
                            color='white', fontsize=6, 
                            bbox=dict(facecolor='black', alpha=0.5))
                    ax1.legend()
                
                # 更新风场数据
                wind_points = f[wind_group_name]['points'][:]
                wind_velocity = f[wind_group_name]['velocity'][:]
                
                # 清除旧的风场图
                ax2.clear()
                im2 = ax2.quiver(wind_points[:, 0], wind_points[:, 1], 
                               wind_velocity[:, 0], wind_velocity[:, 1],
                               scale=50)
                ax2.set_title(f'Wind Field {wind_idx}')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                
                # 在第四个子图中显示数据信息
                ax4.clear()
                ax4.axis('off')
                info_text = f"Data Information:\n"
                info_text += f"HR shape: {hr_data.shape}\n"
                info_text += f"LR shape: {lr_data.shape}\n"
                info_text += f"HR range: [{hr_data.min():.4f}, {hr_data.max():.4f}]\n"
                info_text += f"LR range: [{lr_data.min():.4f}, {lr_data.max():.4f}]\n"
                if 'source_info' in source_group:
                    info_text += f"Source position: ({source_pos[0]/10:.1f}, {source_pos[1]/10:.1f})\n"
                    info_text += f"Source concentration: {source_info[2]:.1f} ppm"
                ax4.text(0.1, 0.5, info_text, fontsize=10, va='center', family='monospace')  # 使用等宽字体
                
            except KeyError as e:
                print(f"无法找到数据: {e}")
                return
            
            fig.canvas.draw_idle()
        
        # 注册更新函数
        wind_slider.on_changed(update)
        source_slider.on_changed(update)
        iter_slider.on_changed(update)
        
        # 添加重置按钮
        reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
        
        def reset(event):
            wind_slider.reset()
            source_slider.reset()
            iter_slider.reset()
        
        reset_button.on_clicked(reset)
        
        # 显示第一个数据
        update(1)
        
        plt.show()

def display_dataset_info(h5_path):
    """
    显示数据集的基本信息
    """
    with h5py.File(h5_path, 'r') as f:
        print("\n数据集信息:")
        for wind_idx in range(1, 4):
            wind_group = f[f'wind{wind_idx}']
            print(f"\n风场 {wind_idx}:")
            print(f"  风场数据:")
            print(f"    位置点数量: {len(wind_group['points'])}")
            print(f"    速度矢量数量: {len(wind_group['velocity'])}")
            print(f"    位置点范围: X[{wind_group['points'][:,0].min():.2f}, {wind_group['points'][:,0].max():.2f}], "
                  f"Y[{wind_group['points'][:,1].min():.2f}, {wind_group['points'][:,1].max():.2f}]")
            print(f"    速度范围: X[{wind_group['velocity'][:,0].min():.2f}, {wind_group['velocity'][:,0].max():.2f}], "
                  f"Y[{wind_group['velocity'][:,1].min():.2f}, {wind_group['velocity'][:,1].max():.2f}]")
            
            for source_idx in range(1, 9):
                source_key = f's{source_idx}'
                if source_key not in wind_group:
                    continue
                source_group = wind_group[source_key]
                print(f"  源位置 {source_idx}:")
                num_keys = [k for k in source_group.keys() if k.startswith('HR_')]
                print(f"    数据集数量: {len(num_keys)}")
                if 'source_info' in source_group:
                    info = source_group['source_info'][:]
                    print(f"    源位置信息: ({info[0]/10:.1f}, {info[1]/10:.1f})")
                    print(f"    源浓度信息: {info[2]:.1f} ppm")
                print(f"    HR数据形状: {source_group['HR_1'].shape}")
                print(f"    LR数据形状: {source_group['LR_1'].shape}")
                print(f"    HR数值范围: [{source_group['HR_1'][:].min():.4f}, {source_group['HR_1'][:].max():.4f}]")
                print(f"    LR数值范围: [{source_group['LR_1'][:].min():.4f}, {source_group['LR_1'][:].max():.4f}]")

def visualize_augmented_data(h5_path):
    """
    Visualize augmented dataset, including original and augmented data
    
    Args:
        h5_path: path to the augmented h5 file
    """
    with h5py.File(h5_path, 'r') as f:
        # 获取所有风场组（包括原始和增强的）
        wind_groups = [k for k in f.keys() if k.startswith('wind')]
        # 获取第一个风场组中的源位置
        source_groups = [k for k in f[wind_groups[0]].keys() if k.startswith('s')]
        
        # 获取第一个数据集的形状
        first_data = f[wind_groups[0]][source_groups[0]]['HR_1'][:]
        
        # 计算实际的迭代次数
        num_iterations = len([k for k in f[wind_groups[0]][source_groups[0]].keys() 
                            if k.startswith('HR_')])
        
        # 创建图形和子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        plt.subplots_adjust(bottom=0.3)
        
        # 创建滑块
        wind_ax = plt.axes([0.2, 0.2, 0.6, 0.03])
        source_ax = plt.axes([0.2, 0.15, 0.6, 0.03])
        iter_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
        
        # 计算实际的风场数量（通过分析wind_groups中的不同wind编号）
        wind_numbers = set()
        for wind_group in wind_groups:
            if '_' in wind_group:
                wind_num = int(wind_group.split('_')[0].replace('wind', ''))
            else:
                wind_num = int(wind_group.replace('wind', ''))
            wind_numbers.add(wind_num)
        num_winds = len(wind_numbers)
        
        wind_slider = Slider(wind_ax, 'Wind', 1, num_winds, valinit=1, valstep=1)
        source_slider = Slider(source_ax, 'Source', 1, len(source_groups), valinit=1, valstep=1)
        iter_slider = Slider(iter_ax, 'Iteration', 1, num_iterations, valinit=1, valstep=1)
        
        # 创建颜色条
        im1 = ax1.imshow(first_data, cmap='viridis')
        cbar1 = plt.colorbar(im1, ax=ax1, label='Concentration')
        
        # 获取风场数据
        wind_points = f[wind_groups[0]]['points'][:]
        wind_velocity = f[wind_groups[0]]['velocity'][:]
        
        # 绘制风场矢量图
        im2 = ax2.quiver(wind_points[:, 0], wind_points[:, 1], 
                        wind_velocity[:, 0], wind_velocity[:, 1],
                        scale=50)
        ax2.set_title('Wind Field')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        # 初始化LR图的显示
        im3 = ax3.imshow(np.zeros((16, 16)), cmap='viridis')
        cbar3 = plt.colorbar(im3, ax=ax3, label='Concentration')
        
        # 创建增强类型按钮
        aug_types = ['Original', 'Rotate 90°', 'Rotate 180°', 'Rotate 270°', 'Flip Horizontal', 'Flip Vertical']
        aug_buttons = []
        button_axes = []
        
        # 为每个增强类型创建按钮
        for i, aug_type in enumerate(aug_types):
            button_ax = plt.axes([0.1 + i*0.15, 0.025, 0.12, 0.04])
            button = Button(button_ax, aug_type, color='lightgoldenrodyellow', hovercolor='0.975')
            aug_buttons.append(button)
            button_axes.append(button_ax)
        
        # 当前选择的增强类型
        current_aug_type = 0
        
        def update(val):
            wind_idx = int(wind_slider.val)
            source_idx = int(source_slider.val)
            iter_idx = int(iter_slider.val)
            
            # 根据当前增强类型选择风场组
            wind_base = f'wind{wind_idx}'
            if current_aug_type == 0:
                wind_group_name = f'{wind_base}_0'
            elif current_aug_type == 1:
                wind_group_name = f'{wind_base}_rot90'
            elif current_aug_type == 2:
                wind_group_name = f'{wind_base}_rot180'
            elif current_aug_type == 3:
                wind_group_name = f'{wind_base}_rot270'
            elif current_aug_type == 4:
                wind_group_name = f'{wind_base}_flip_h'
            else:  # current_aug_type == 5
                wind_group_name = f'{wind_base}_flip_v'
            
            source_group_name = f's{source_idx}'
            
            try:
                # 更新HR浓度数据
                wind_group = f[wind_group_name]
                source_group = wind_group[source_group_name]
                hr_data = source_group[f'HR_{iter_idx}'][:]
                lr_data = source_group[f'LR_{iter_idx}'][:]
                
                # 更新HR图
                im1.set_data(hr_data)
                im1.set_clim(vmin=hr_data.min(), vmax=hr_data.max())
                ax1.set_title(f'HR Concentration - {wind_group_name}, Source {source_idx}, Iteration {iter_idx}')
                
                # 更新LR图
                im3.set_data(lr_data)
                im3.set_clim(vmin=lr_data.min(), vmax=lr_data.max())
                ax3.set_title(f'LR Concentration (16x16)')
                
                # 清除之前的所有标记和文本
                for artist in ax1.lines + ax1.texts:
                    artist.remove()
                if ax1.legend_:
                    ax1.legend_.remove()
                
                if 'source_info' in source_group:
                    source_info = source_group['source_info'][:]
                    source_pos = source_info[:2]
                    ax1.plot(source_pos[0], source_pos[1], 'r*', markersize=5, label='Source')
                    ax1.text(source_pos[0]+5, source_pos[1]+5, 
                            f'({source_pos[0]/10:.1f}, {source_pos[1]/10:.1f})',
                            color='white', fontsize=6, 
                            bbox=dict(facecolor='black', alpha=0.5))
                    ax1.legend()
                
                # 更新风场数据
                wind_points = wind_group['points'][:]
                wind_velocity = wind_group['velocity'][:]
                
                # 清除旧的风场图
                ax2.clear()
                im2 = ax2.quiver(wind_points[:, 0], wind_points[:, 1], 
                               wind_velocity[:, 0], wind_velocity[:, 1],
                               scale=50)
                ax2.set_title(f'Wind Field - {wind_group_name}')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                
                # 在第四个子图中显示数据信息
                ax4.clear()
                ax4.axis('off')
                info_text = f"Data Information:\n"
                info_text += f"Wind Group: {wind_group_name}\n"
                info_text += f"Augmentation Type: {aug_types[current_aug_type]}\n"
                info_text += f"HR shape: {hr_data.shape}\n"
                info_text += f"LR shape: {lr_data.shape}\n"
                info_text += f"HR range: [{hr_data.min():.4f}, {hr_data.max():.4f}]\n"
                info_text += f"LR range: [{lr_data.min():.4f}, {lr_data.max():.4f}]\n"
                if 'source_info' in source_group:
                    info_text += f"Source position: ({source_pos[0]/10:.1f}, {source_pos[1]/10:.1f})\n"
                    info_text += f"Source concentration: {source_info[2]:.1f} ppm"
                ax4.text(0.1, 0.5, info_text, fontsize=10, va='center', family='monospace')
                
            except KeyError as e:
                print(f"无法找到数据: {e}")
                return
            
            fig.canvas.draw_idle()
        
        # 创建按钮回调函数
        def create_button_callback(i):
            def callback(event):
                nonlocal current_aug_type
                current_aug_type = i
                # 更新按钮颜色
                for j, button in enumerate(aug_buttons):
                    button.color = 'lightgoldenrodyellow' if j != i else 'lightblue'
                update(1)
            return callback
        
        # 注册按钮回调
        for i, button in enumerate(aug_buttons):
            button.on_clicked(create_button_callback(i))
        
        # 注册滑块更新函数
        wind_slider.on_changed(update)
        source_slider.on_changed(update)
        iter_slider.on_changed(update)
        
        # 添加重置按钮
        reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
        
        def reset(event):
            wind_slider.reset()
            source_slider.reset()
            iter_slider.reset()
            # 重置增强类型按钮
            nonlocal current_aug_type
            current_aug_type = 0
            for i, button in enumerate(aug_buttons):
                button.color = 'lightgoldenrodyellow' if i != 0 else 'lightblue'
            update(1)
        
        reset_button.on_clicked(reset)
        
        # 显示第一个数据
        update(1)
        
        plt.show()

def display_augmented_dataset_info(h5_path):
    """
    显示增强数据集的基本信息
    """
    with h5py.File(h5_path, 'r') as f:
        print("\n增强数据集信息:")
        for wind_group_name in f.keys():
            wind_group = f[wind_group_name]
            print(f"\n风场组 {wind_group_name}:")
            print(f"  风场数据:")
            print(f"    位置点数量: {len(wind_group['points'])}")
            print(f"    速度矢量数量: {len(wind_group['velocity'])}")
            print(f"    位置点范围: X[{wind_group['points'][:,0].min():.2f}, {wind_group['points'][:,0].max():.2f}], "
                  f"Y[{wind_group['points'][:,1].min():.2f}, {wind_group['points'][:,1].max():.2f}]")
            print(f"    速度范围: X[{wind_group['velocity'][:,0].min():.2f}, {wind_group['velocity'][:,0].max():.2f}], "
                  f"Y[{wind_group['velocity'][:,1].min():.2f}, {wind_group['velocity'][:,1].max():.2f}]")
            
            for source_idx in range(1, 9):
                source_key = f's{source_idx}'
                if source_key not in wind_group:
                    continue
                source_group = wind_group[source_key]
                print(f"  源位置 {source_idx}:")
                num_keys = [k for k in source_group.keys() if k.startswith('HR_')]
                print(f"    数据集数量: {len(num_keys)}")
                if 'source_info' in source_group:
                    info = source_group['source_info'][:]
                    print(f"    源位置信息: ({info[0]/10:.1f}, {info[1]/10:.1f})")
                    print(f"    源浓度信息: {info[2]:.1f} ppm")
                print(f"    HR数据形状: {source_group['HR_1'].shape}")
                print(f"    LR数据形状: {source_group['LR_1'].shape}")
                print(f"    HR数值范围: [{source_group['HR_1'][:].min():.4f}, {source_group['HR_1'][:].max():.4f}]")
                print(f"    LR数值范围: [{source_group['LR_1'][:].min():.4f}, {source_group['LR_1'][:].max():.4f}]")

def main():
    # 让用户选择要可视化的文件
    print("请选择要可视化的文件：")
    print("1. 原始数据集 (dataset.h5)")
    print("2. 增强数据集 (augmented_dataset.h5)")
    print("3. 自定义文件路径")
    
    choice = input("请输入选项（1/2/3）：")
    
    if choice == '1':
        h5_path = 'C:\\Users\\yy143\\Desktop\\dataset\\source_dataset\\test_dataset.h5'  # 修复缩进
    elif choice == '2':
        h5_path = 'C:\\Users\\yy143\\Desktop\\dataset\\source_dataset\\test_normalized_augmented_dataset.h5'
    elif choice == '3':
        h5_path = input("请输入h5文件的完整路径：")
    else:
        print("无效的选项！")
        return
    
    # 检查文件是否存在
    if not os.path.exists(h5_path):
        print(f"文件 {h5_path} 不存在！")
        return
    
    # 检查文件是否为h5文件
    if not h5_path.endswith('.h5'):
        print("请选择.h5文件！")
        return
    
    try:
        # 尝试打开文件并检查数据结构
        with h5py.File(h5_path, 'r') as f:
            # 检查是否为增强数据集（通过检查wind组名是否包含_0, _rot90等后缀）
            is_augmented = any('_' in key for key in f.keys())
            
            if is_augmented:
                print("检测到增强数据集，使用增强数据集可视化...")
                display_augmented_dataset_info(h5_path)
                visualize_augmented_data(h5_path)
            else:
                print("检测到原始数据集，使用原始数据集可视化...")
                display_dataset_info(h5_path)  # 修复缩进
                visualize_concentration_and_wind(h5_path)  # 修复缩进
    except Exception as e:
        print(f"打开文件时出错：{e}")
        return

if __name__ == '__main__':
    main()
