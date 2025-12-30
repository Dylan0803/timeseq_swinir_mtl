import os
import json
import matplotlib.pyplot as plt


def plot_loss_curve(losses, save_path=None):
    """绘制训练损失曲线"""
    plt.figure()
    plt.plot(range(len(losses)), losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def save_args(args, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')


def generate_readme(args, result_msg):
    """生成 ReadMe.txt，保存训练摘要信息"""
    readme_file = os.path.join(args.output_dir, 'ReadMe.txt')
    with open(readme_file, 'w') as f:
        lines = []
        lines.append(f'Train and valid with {args.input_file}\n')
        lines.append(f'Model: {args.model_name}\n')
        lines.append(result_msg)

        for line in lines:
            print(line)

        f.writelines(lines)


def get_last_net_dir(models_dir, model_name):
    """获取模型保存目录下最新的模型文件夹路径"""
    model_param_path = os.path.join(models_dir, model_name)
    model_save_time = None
    model_save_time = model_save_time \
        if model_save_time is not None else \
        sorted(
            list(filter(os.path.isdir, [os.path.join(
                model_param_path, x) for x in os.listdir(model_param_path)])),
            reverse=True)[3]
    return model_save_time


def plot_loss_lines(args, train_losses, valid_losses):
    """绘制训练与验证损失图"""
    import matplotlib.pyplot as plt
    import os

    # 使用默认字体，不使用SimHei
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # 确保基础目录存在
        base_dir = './experiments'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"Created base directory: {base_dir}")

        # 确保模型目录存在
        model_dir = os.path.join(base_dir, args.model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created model directory: {model_dir}")

        # 确保实验目录存在
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print(f"Created experiment directory: {args.output_dir}")

        # ✅ 确保损失图目录存在
        loss_dir = os.path.join(model_dir, 'loss_plots')
        os.makedirs(loss_dir, exist_ok=True)

        # train loss
        figure_train = plt.figure()
        plt.plot(range(len(train_losses)), train_losses, color='r', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(loss_dir, 'Train-Loss.png')
        figure_train.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved training loss plot to: {save_path}")

        # valid loss
        figure_valid = plt.figure()
        plt.plot(range(len(valid_losses)), valid_losses, color='b', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(loss_dir, 'Valid-Loss.png')
        figure_valid.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved validation loss plot to: {save_path}")

        # Combined plot
        figure_combined = plt.figure()
        plt.plot(range(len(train_losses)), train_losses, color='r', label='Training Loss')
        plt.plot(range(len(valid_losses)), valid_losses, color='b', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(loss_dir, 'Combined-Loss.png')
        figure_combined.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved combined loss plot to: {save_path}")

    except Exception as e:
        print(f"Error occurred while plotting loss curves: {str(e)}")
        plt.close('all')


if __name__ == '__main__':
    pass
