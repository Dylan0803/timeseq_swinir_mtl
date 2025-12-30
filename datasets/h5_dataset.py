"""
多任务学习数据集加载器，用于超分辨率重建和泄漏源位置预测
"""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random

# 设置随机种子


def set_seed(seed=42):
    """
    设置随机种子，确保结果可复现

    参数：
    seed: 随机种子值，默认42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MultiTaskDataset(Dataset):
    def __init__(self, dataset_file_name, index_list=None, shuffle=True):
        """
        参数：
        dataset_file_name：.h5 文件路径
        index_list：用于指定使用哪些索引，默认使用全部
        shuffle：是否在 index_list 内部进行打乱
        """
        super(MultiTaskDataset, self).__init__()
        self.dataset_file = h5py.File(dataset_file_name, 'r')

        # 构建数据索引列表
        self.data_indices = []

        # 遍历所有风场组
        for wind_group_name in self.dataset_file.keys():
            wind_group = self.dataset_file[wind_group_name]
            # 获取所有源位置组（s1-s8）
            source_groups = [f's{i}' for i in range(1, 9)]

            # 遍历所有源位置
            for source_group_name in source_groups:
                try:
                    source_group = wind_group[source_group_name]
                    # 获取时间步数量
                    time_steps = len(
                        [k for k in source_group.keys() if k.startswith('HR_')])
                    # 为每个时间步创建索引
                    for time_step in range(1, time_steps + 1):
                        self.data_indices.append({
                            'wind_group': wind_group_name,
                            'source_group': source_group_name,
                            'time_step': time_step
                        })
                except Exception:
                    continue

        # 如果未指定索引列表，就使用全部数据
        if index_list is None:
            index_list = list(range(len(self.data_indices)))

        # 是否打乱索引
        if shuffle:
            random.shuffle(index_list)

        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        """
        返回数据时，自动读取 LR、HR 和泄漏源位置信息，并进行归一化处理
        返回：
        - lr_tensor: 低分辨率数据 [1, H, W]，值范围[0,1]
        - hr_tensor: 高分辨率数据 [1, H, W]，值范围[0,1]
        - source_pos: 泄漏源位置真值 [2]，值范围[0,1]
        - hr_max_pos: HR中浓度最高值的位置 [2]，值范围[0,1]
        - wind_vector: 风场向量 [2]，值范围[-1,1]
        """
        try:
            idx_in_file = self.index_list[idx]
            data_info = self.data_indices[idx_in_file]

            # 获取数据组
            wind_group = self.dataset_file[data_info['wind_group']]
            source_group = wind_group[data_info['source_group']]

            # 读取LR和HR数据
            lr = source_group[f'LR_{data_info["time_step"]}'][:]
            hr = source_group[f'HR_{data_info["time_step"]}'][:]

            # 读取泄漏源位置信息（前两个值是x, y坐标）
            source_info = source_group['source_info'][:]
            source_pos = source_info[:2]  # 只取位置信息

            # 计算HR中浓度最高值的位置
            hr_max_pos = np.unravel_index(hr.argmax(), hr.shape)
            hr_max_pos = np.array(
                # 转换为(x, y)顺序
                [hr_max_pos[1], hr_max_pos[0]], dtype=np.float32)

            # 获取图像尺寸
            _, width = hr.shape
            # 归一化坐标到[0,1]范围
            source_pos = source_pos / (width - 1)
            hr_max_pos = hr_max_pos / (width - 1)

            # 读取风场信息
            wind_velocity = wind_group['velocity'][:]  # 读取风场速度
            # 归一化风场向量到[-1,1]范围
            wind_vector = wind_velocity / np.max(np.abs(wind_velocity))

            # 增加通道维度
            if len(lr.shape) == 2:
                lr = lr[np.newaxis, :, :]
            if len(hr.shape) == 2:
                hr = hr[np.newaxis, :, :]

            # 转换为tensor
            lr_tensor = torch.tensor(lr, dtype=torch.float32)
            hr_tensor = torch.tensor(hr, dtype=torch.float32)
            source_pos_tensor = torch.tensor(source_pos, dtype=torch.float32)
            hr_max_pos_tensor = torch.tensor(hr_max_pos, dtype=torch.float32)
            wind_vector_tensor = torch.tensor(wind_vector, dtype=torch.float32)

            return {
                'lr': lr_tensor,          # 输入数据，已归一化到[0,1]
                'hr': hr_tensor,          # 超分辨率任务的目标，已归一化到[0,1]
                'source_pos': source_pos_tensor,  # 泄漏源位置真值，已归一化到[0,1]
                'hr_max_pos': hr_max_pos_tensor,  # HR中浓度最高位置，已归一化到[0,1]
                'wind_vector': wind_vector_tensor  # 风场向量，已归一化到[-1,1]
            }
        except Exception as e:
            print(f"错误：处理索引 {idx} 时发生错误: {str(e)}")
            raise


class MultiTaskSeqDataset(Dataset):
    """
    时间序列多任务数据集加载器

    用于训练两个任务：
    1. 根据历史 K 帧 LR 序列，重建当前 HR_t（SR@t）
    2. 根据历史 K 帧 LR 序列，预测下一帧 HR_{t+1}（Pred@t+1）

    输出字典包含：
    - lr_seq: shape = (K, 1, h_lr, w_lr), float32
    - hr_t: shape = (1, h_hr, w_hr), float32
    - hr_tp1: shape = (1, h_hr, w_hr), float32
    - source_pos: shape = (2,), float32 [0,1]
    - hr_max_pos_t: shape = (2,), float32 [0,1]
    - hr_max_pos_tp1: shape = (2,), float32 [0,1]
    - wind_vector: shape = (2,), float32 [-1,1]
    """

    def __init__(self, dataset_file_name, index_list=None, K=6, shuffle=True, lazy_open=False, return_single=False):
        """
        参数：
        dataset_file_name：.h5 文件路径
        index_list：用于指定使用哪些索引，默认使用全部
        K：历史帧数，默认6
        shuffle：是否在 index_list 内部进行打乱
        lazy_open：是否延迟打开文件（用于多进程 DataLoader，建议设置 num_workers=0 来避免 h5py 线程安全问题）
        return_single：是否只返回单帧（调试用），默认 False
        """
        super(MultiTaskSeqDataset, self).__init__()
        self.K = K
        self.lazy_open = lazy_open
        self.return_single = return_single
        self.dataset_file_name = dataset_file_name

        if not lazy_open:
            self.dataset_file = h5py.File(dataset_file_name, 'r')
        else:
            self.dataset_file = None

        # 构建数据索引列表
        # 每个样本是中心 time_step=t，对应历史窗口 [t-K+1, ..., t]，需要 HR_{t+1}
        # 合法 t 范围：K 到 T-1（包含）
        self.data_indices = []

        # 临时打开文件以构建索引
        with h5py.File(dataset_file_name, 'r') as f:
            # 遍历所有风场组
            for wind_group_name in f.keys():
                wind_group = f[wind_group_name]
                # 获取所有源位置组（s1-s8）
                source_groups = [f's{i}' for i in range(1, 9)]

                # 遍历所有源位置
                for source_group_name in source_groups:
                    try:
                        source_group = wind_group[source_group_name]
                        # 获取时间步数量（通过统计 HR_ keys）
                        hr_keys = [k for k in source_group.keys()
                                   if k.startswith('HR_')]
                        T = len(hr_keys)

                        # 检查是否满足条件：T >= K+1
                        if T < K + 1:
                            print(
                                f"警告：跳过 {wind_group_name}/{source_group_name}，时间步数 T={T} < K+1={K+1}")
                            continue

                        # 为每个合法的中心时间步 t 创建索引
                        # t 从 K 到 T-1（包含）
                        for t in range(K, T):
                            self.data_indices.append({
                                'wind_group': wind_group_name,
                                'source_group': source_group_name,
                                't': t  # 1-indexed 中心时间步
                            })
                    except Exception as e:
                        print(
                            f"警告：处理 {wind_group_name}/{source_group_name} 时出错: {str(e)}")
                        continue

        # 如果未指定索引列表，就使用全部数据
        if index_list is None:
            index_list = list(range(len(self.data_indices)))

        # 是否打乱索引
        if shuffle:
            random.shuffle(index_list)

        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def _get_file_handle(self):
        """获取文件句柄，支持延迟打开"""
        if self.lazy_open:
            if self.dataset_file is None:
                self.dataset_file = h5py.File(self.dataset_file_name, 'r')
            return self.dataset_file
        else:
            return self.dataset_file

    def __getitem__(self, idx):
        """
        返回时序数据

        返回字典：
        - lr_seq: (K, 1, h_lr, w_lr), float32 - 历史 K 帧 LR 序列
        - hr_t: (1, h_hr, w_hr), float32 - 当前帧 HR
        - hr_tp1: (1, h_hr, w_hr), float32 - 下一帧 HR
        - source_pos: (2,), float32 [0,1] - 泄漏源位置
        - hr_max_pos_t: (2,), float32 [0,1] - HR_t 的最大值位置
        - hr_max_pos_tp1: (2,), float32 [0,1] - HR_{t+1} 的最大值位置
        - wind_vector: (2,), float32 [-1,1] - 风场向量
        """
        try:
            idx_in_file = self.index_list[idx]
            data_info = self.data_indices[idx_in_file]

            # 获取文件句柄
            f = self._get_file_handle()

            # 获取数据组
            wind_group = f[data_info['wind_group']]
            source_group = wind_group[data_info['source_group']]
            t = data_info['t']  # 中心时间步（1-indexed）

            # 读取历史 K 帧 LR 序列：LR_{t-K+1} ... LR_t
            lr_seq_list = []
            for i in range(self.K):
                time_step = t - self.K + 1 + i  # 1-indexed: 从 t-K+1 到 t
                lr_key = f'LR_{time_step}'
                if lr_key not in source_group:
                    raise KeyError(
                        f"缺少键 {lr_key} in {data_info['wind_group']}/{data_info['source_group']}")
                lr_frame = source_group[lr_key][:]
                lr_seq_list.append(lr_frame)

            # 堆叠成序列，时间维在第0维
            lr_seq = np.stack(lr_seq_list, axis=0)  # shape: (K, h_lr, w_lr)

            # 读取当前帧 HR_t
            hr_key_t = f'HR_{t}'
            if hr_key_t not in source_group:
                raise KeyError(
                    f"缺少键 {hr_key_t} in {data_info['wind_group']}/{data_info['source_group']}")
            hr_t = source_group[hr_key_t][:]  # shape: (h_hr, w_hr)

            # 读取下一帧 HR_{t+1}
            hr_key_tp1 = f'HR_{t+1}'
            if hr_key_tp1 not in source_group:
                raise KeyError(
                    f"缺少键 {hr_key_tp1} in {data_info['wind_group']}/{data_info['source_group']}")
            hr_tp1 = source_group[hr_key_tp1][:]  # shape: (h_hr, w_hr)

            # 读取泄漏源位置信息（前两个值是x, y坐标）
            if 'source_info' not in source_group:
                raise KeyError(
                    f"缺少键 source_info in {data_info['wind_group']}/{data_info['source_group']}")
            source_info = source_group['source_info'][:]
            source_pos = source_info[:2]  # 只取位置信息

            # 计算 HR_t 中浓度最高值的位置
            hr_max_pos_t = np.unravel_index(hr_t.argmax(), hr_t.shape)
            hr_max_pos_t = np.array(
                # 转换为(x, y)顺序
                [hr_max_pos_t[1], hr_max_pos_t[0]], dtype=np.float32)

            # 计算 HR_{t+1} 中浓度最高值的位置
            hr_max_pos_tp1 = np.unravel_index(hr_tp1.argmax(), hr_tp1.shape)
            hr_max_pos_tp1 = np.array(
                # 转换为(x, y)顺序
                [hr_max_pos_tp1[1], hr_max_pos_tp1[0]], dtype=np.float32)

            # 获取 HR 图像尺寸（用于归一化）
            _, width = hr_t.shape
            # 归一化坐标到[0,1]范围（注意使用 HR 的宽度）
            source_pos = source_pos / (width - 1)
            hr_max_pos_t = hr_max_pos_t / (width - 1)
            hr_max_pos_tp1 = hr_max_pos_tp1 / (width - 1)

            # 读取风场信息
            if 'velocity' not in wind_group:
                raise KeyError(f"缺少键 velocity in {data_info['wind_group']}")
            wind_velocity = wind_group['velocity'][:]  # 读取风场速度
            # 归一化风场向量到[-1,1]范围
            wind_vector = wind_velocity / np.max(np.abs(wind_velocity))

            # 增加通道维度
            # lr_seq: (K, h_lr, w_lr) -> (K, 1, h_lr, w_lr)
            if len(lr_seq.shape) == 3:
                lr_seq = lr_seq[:, np.newaxis, :, :]
            # hr_t: (h_hr, w_hr) -> (1, h_hr, w_hr)
            if len(hr_t.shape) == 2:
                hr_t = hr_t[np.newaxis, :, :]
            # hr_tp1: (h_hr, w_hr) -> (1, h_hr, w_hr)
            if len(hr_tp1.shape) == 2:
                hr_tp1 = hr_tp1[np.newaxis, :, :]

            # 转换为 tensor
            lr_seq_tensor = torch.tensor(lr_seq, dtype=torch.float32)
            hr_t_tensor = torch.tensor(hr_t, dtype=torch.float32)
            hr_tp1_tensor = torch.tensor(hr_tp1, dtype=torch.float32)
            source_pos_tensor = torch.tensor(source_pos, dtype=torch.float32)
            hr_max_pos_t_tensor = torch.tensor(
                hr_max_pos_t, dtype=torch.float32)
            hr_max_pos_tp1_tensor = torch.tensor(
                hr_max_pos_tp1, dtype=torch.float32)
            wind_vector_tensor = torch.tensor(wind_vector, dtype=torch.float32)

            result = {
                # 历史 K 帧 LR 序列，shape: (K, 1, h_lr, w_lr)
                'lr_seq': lr_seq_tensor,
                # 当前帧 HR，shape: (1, h_hr, w_hr)
                'hr_t': hr_t_tensor,
                # 下一帧 HR，shape: (1, h_hr, w_hr)
                'hr_tp1': hr_tp1_tensor,
                'source_pos': source_pos_tensor,  # 泄漏源位置真值，shape: (2,)
                # HR_t 中浓度最高位置，shape: (2,)
                'hr_max_pos_t': hr_max_pos_t_tensor,
                # HR_{t+1} 中浓度最高位置，shape: (2,)
                'hr_max_pos_tp1': hr_max_pos_tp1_tensor,
                'wind_vector': wind_vector_tensor  # 风场向量，shape: (2,)
            }

            # 如果 return_single=True，只返回单帧（调试用）
            if self.return_single:
                result['lr'] = lr_seq_tensor[-1]  # 只返回最后一帧 LR
                result['hr'] = hr_t_tensor  # 返回当前帧 HR

            return result

        except Exception as e:
            print(f"错误：处理索引 {idx} (wind={data_info.get('wind_group', 'unknown')}, "
                  f"source={data_info.get('source_group', 'unknown')}, "
                  f"t={data_info.get('t', 'unknown')}) 时发生错误: {str(e)}")
            raise

    def __del__(self):
        """关闭文件句柄"""
        if hasattr(self, 'dataset_file') and self.dataset_file is not None:
            try:
                self.dataset_file.close()
            except Exception:
                pass


def generate_train_valid_test_dataset(data_file, train_ratio=0.8, valid_ratio=0.1, shuffle=True, seed=42, K=6, use_seq_dataset=False):
    """
    生成训练集、验证集和测试集 (按泄漏模拟组划分，防止数据泄露)

    参数：
    data_file：.h5 文件路径
    train_ratio：训练集比例，默认0.8
    valid_ratio：验证集比例，默认0.1
    shuffle：是否打乱模拟组，默认True
    seed：随机种子，默认42
    K：历史帧数（仅用于时序数据集），默认6
    use_seq_dataset：是否使用时序数据集 MultiTaskSeqDataset，默认False（使用 MultiTaskDataset）
    """
    set_seed(seed)

    # 1. 识别所有独立的模拟组
    simulation_groups = []
    with h5py.File(data_file, 'r') as f:
        for wind_group_name in f.keys():
            wind_group = f[wind_group_name]
            source_groups_in_file = [f's{i}' for i in range(1, 9)]
            for source_group_name in source_groups_in_file:
                if source_group_name in wind_group:
                    simulation_groups.append(
                        (wind_group_name, source_group_name))

    # 2. 打乱模拟组
    if shuffle:
        random.shuffle(simulation_groups)

    # 3. 按比例划分
    num_groups = len(simulation_groups)
    train_split_idx = int(train_ratio * num_groups)
    valid_split_idx = int((train_ratio + valid_ratio) * num_groups)
    train_groups = simulation_groups[:train_split_idx]
    valid_groups = simulation_groups[train_split_idx:valid_split_idx]
    test_groups = simulation_groups[valid_split_idx:]
    print(f"总共有 {num_groups} 个独立的模拟组。")
    print(
        f"划分: {len(train_groups)} 组用于训练, {len(valid_groups)} 组用于验证, {len(test_groups)} 组用于测试。")

    if use_seq_dataset:
        # 时序数据集：索引构建方式不同
        # 每个样本是中心 time_step=t，需要历史 K 帧和 HR_{t+1}
        # 合法 t 范围：K 到 T-1（包含）

        # 4. 构建全局索引映射（时序数据集）
        all_data_indices = []
        global_idx_map = {}
        with h5py.File(data_file, 'r') as f:
            current_global_idx = 0
            for wind_group_name in f.keys():
                wind_group = f[wind_group_name]
                source_groups_in_file = [f's{i}' for i in range(1, 9)]
                for source_group_name in source_groups_in_file:
                    if source_group_name in wind_group:
                        source_group = wind_group[source_group_name]
                        hr_keys = [k for k in source_group.keys()
                                   if k.startswith('HR_')]
                        T = len(hr_keys)

                        # 检查是否满足条件：T >= K+1
                        if T < K + 1:
                            print(
                                f"警告：跳过 {wind_group_name}/{source_group_name}，时间步数 T={T} < K+1={K+1}")
                            continue

                        # 遍历合法的中心时间步 t：从 K 到 T-1（包含）
                        for t in range(K, T):
                            all_data_indices.append({
                                'wind_group': wind_group_name,
                                'source_group': source_group_name,
                                't': t  # 1-indexed 中心时间步
                            })
                            key = (wind_group_name, source_group_name, t)
                            global_idx_map[key] = current_global_idx
                            current_global_idx += 1

        # 5. 根据组划分生成索引列表（时序数据集）
        train_list, valid_list, test_list = [], [], []
        with h5py.File(data_file, 'r') as f:
            for group_set, index_list in [(train_groups, train_list), (valid_groups, valid_list), (test_groups, test_list)]:
                for wind_group_name, source_group_name in group_set:
                    if source_group_name not in f[wind_group_name]:
                        continue
                    source_group = f[wind_group_name][source_group_name]
                    hr_keys = [k for k in source_group.keys()
                               if k.startswith('HR_')]
                    T = len(hr_keys)

                    # 检查是否满足条件
                    if T < K + 1:
                        continue

                    # 遍历合法的中心时间步 t
                    for t in range(K, T):
                        key = (wind_group_name, source_group_name, t)
                        if key in global_idx_map:
                            index_list.append(global_idx_map[key])

        # 6. 创建时序数据集实例
        train_dataset = MultiTaskSeqDataset(
            data_file, train_list, K=K, shuffle=True)
        valid_dataset = MultiTaskSeqDataset(
            data_file, valid_list, K=K, shuffle=False)
        test_dataset = MultiTaskSeqDataset(
            data_file, test_list, K=K, shuffle=False)

    else:
        # 原始数据集：索引构建方式
        # 4. 构建全局索引映射（原始数据集）
        all_data_indices = []
        global_idx_map = {}
        with h5py.File(data_file, 'r') as f:
            current_global_idx = 0
            for wind_group_name in f.keys():
                wind_group = f[wind_group_name]
                source_groups_in_file = [f's{i}' for i in range(1, 9)]
                for source_group_name in source_groups_in_file:
                    if source_group_name in wind_group:
                        source_group = wind_group[source_group_name]
                        time_steps_count = len(
                            [k for k in source_group.keys() if k.startswith('HR_')])
                        for time_step in range(1, time_steps_count + 1):
                            all_data_indices.append({
                                'wind_group': wind_group_name,
                                'source_group': source_group_name,
                                'time_step': time_step
                            })
                            key = (wind_group_name,
                                   source_group_name, time_step)
                            global_idx_map[key] = current_global_idx
                            current_global_idx += 1

        # 5. 根据组划分生成索引列表（原始数据集）
        train_list, valid_list, test_list = [], [], []
        with h5py.File(data_file, 'r') as f:
            for group_set, index_list in [(train_groups, train_list), (valid_groups, valid_list), (test_groups, test_list)]:
                for wind_group_name, source_group_name in group_set:
                    source_group = f[wind_group_name][source_group_name]
                    time_steps_count = len(
                        [k for k in source_group.keys() if k.startswith('HR_')])
                    for time_step in range(1, time_steps_count + 1):
                        key = (wind_group_name, source_group_name, time_step)
                        index_list.append(global_idx_map[key])

        # 6. 创建原始数据集实例
        train_dataset = MultiTaskDataset(data_file, train_list, shuffle=True)
        valid_dataset = MultiTaskDataset(data_file, valid_list, shuffle=False)
        test_dataset = MultiTaskDataset(data_file, test_list, shuffle=False)

    return train_dataset, valid_dataset, test_dataset


# ================== 泄漏检查测试模块 ==================
if __name__ == "__main__":
    # 假设你有 MultiTaskDataset 类和一个 h5 文件路径
    data_file = "your_data.h5"
    train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
        data_file)

    # 提取每个数据集用到的 simulation_groups
    def get_sim_groups(dataset):
        sim_groups = set()
        for idx in dataset.indices:
            info = dataset.all_data_indices[idx]
            sim_groups.add((info['wind_group'], info['source_group']))
        return sim_groups

    train_groups = get_sim_groups(train_dataset)
    test_groups = get_sim_groups(test_dataset)
    valid_groups = get_sim_groups(valid_dataset)

    print("训练集与测试集的组交集：", train_groups & test_groups)
    print("训练集与验证集的组交集：", train_groups & valid_groups)
    print("验证集与测试集的组交集：", valid_groups & test_groups)
    if not (train_groups & test_groups):
        print("✅ 训练集与测试集无泄漏！")
    else:
        print("❌ 训练集与测试集有泄漏！")
