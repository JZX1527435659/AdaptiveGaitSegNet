import os
import os.path as osp
import numpy as np

from .data_set import DataSet


def _collect_sequences(root_dir: str, data_type: str, direction: str):
    seq_dir = []
    label = []
    seq_type = []
    view = []

    if not osp.isdir(root_dir):
        raise ValueError(f"数据集路径不存在: {root_dir}")

    gait_groups = ["pre_normal", "pre_parkinsonian"]
    for gait_group in gait_groups:
        group_path = osp.join(root_dir, gait_group)
        if not osp.isdir(group_path):
            # 允许缺失某一组
            continue

        subjects = [d for d in sorted(os.listdir(group_path)) if d.startswith("sub")]
        for subject in subjects:
            subject_path = osp.join(group_path, subject)
            # 选择 silhouettes 或 GEIs
            folder_name = "GEIs" if data_type.lower() == "gei" else "silhouettes"
            target_path = osp.join(subject_path, folder_name)
            if not osp.isdir(target_path):
                continue

            # 遍历序列文件夹，筛选方向
            for seq_folder in sorted(os.listdir(target_path)):
                if direction.lower() not in seq_folder.lower():
                    continue
                seq_path = osp.join(target_path, seq_folder)
                if not osp.isdir(seq_path):
                    continue

                
                frames = [f for f in os.listdir(seq_path) if f.lower().endswith(('.png', '.jpg'))]
                if len(frames) == 0:
                    continue

                # 标签: 正常=0, 帕金森=1
                is_parkinson = 1 if gait_group == "pre_parkinsonian" else 0
                label.append(str(is_parkinson))
                view.append(direction)
                seq_type.append(data_type)
                seq_dir.append([seq_path])

    return seq_dir, label, seq_type, view


def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True, data_type='silhouettes', direction='front', balance=False):
    """
    适配 GAIT-IST（预处理后）的数据加载。

    - dataset_path: 指向 gaitist_output
    - data_type: 'silhouettes' 或 'GEIs'
    - direction: 'front' 或 'back'
    - balance: True=类均衡采样; False=使用全部样本(数量更多)
    - 以二分类(正常0/帕金森1)组织 label
    - 返回 DataSet 以兼容原训练/测试流程
    """

    seq_dir, label, seq_type, view = _collect_sequences(dataset_path, data_type, direction)

    if len(seq_dir) == 0:
        raise ValueError(f"未在 {dataset_path} 中找到 {data_type} 且方向为 {direction} 的样本")

    # 使用全部样本进行划分（80/20）
    normal_idx = [i for i, l in enumerate(label) if l == '0']
    park_idx = [i for i, l in enumerate(label) if l == '1']
    if len(normal_idx) == 0 or len(park_idx) == 0:
        raise ValueError("二分类需要同时包含正常与帕金森样本，请检查数据集")

    # 随机划分训练/测试（80/20），根据balance参数决定是否使用均衡采样
    rng = np.random.RandomState(0 if not pid_shuffle else 42)
    
    if balance:
        # 类均衡采样：使两个类别的样本数相等
        min_count = min(len(normal_idx), len(park_idx))
        # 随机选择相等数量的样本
        rng.shuffle(normal_idx)
        rng.shuffle(park_idx)
        normal_idx = normal_idx[:min_count]
        park_idx = park_idx[:min_count]
        print(f"⚖️ 使用类均衡采样，每个类别样本数: {min_count}")
    
    # 对Normal样本进行8:2划分
    rng.shuffle(normal_idx)
    normal_split = max(1, int(len(normal_idx) * 0.8))
    normal_train_ids = normal_idx[:normal_split]
    normal_test_ids = normal_idx[normal_split:]
    
    # 对Parkinson样本进行8:2划分
    rng.shuffle(park_idx)
    park_split = max(1, int(len(park_idx) * 0.8))
    park_train_ids = park_idx[:park_split]
    park_test_ids = park_idx[park_split:]
    
    # 合并训练和测试样本
    train_ids = np.array(normal_train_ids + park_train_ids)
    test_ids = np.array(normal_test_ids + park_test_ids)
    
    # 随机打乱训练和测试集
    rng.shuffle(train_ids)
    rng.shuffle(test_ids)

    # 打印样本统计信息
    print(f"📊 数据集统计信息:")
    print(f"   总样本数: {len(seq_dir)}")
    print(f"   Normal样本数: {len(normal_idx)}")
    print(f"   Parkinson样本数: {len(park_idx)}")
    
    # 计算训练集和测试集中各类别的样本数
    train_normal_count = sum(1 for idx in train_ids if label[idx] == '0')
    train_parkinson_count = sum(1 for idx in train_ids if label[idx] == '1')
    test_normal_count = sum(1 for idx in test_ids if label[idx] == '0')
    test_parkinson_count = sum(1 for idx in test_ids if label[idx] == '1')
    
    print(f"   训练集样本数: {len(train_ids)} (Normal: {train_normal_count}, Parkinson: {train_parkinson_count})")
    print(f"   测试集样本数: {len(test_ids)} (Normal: {test_normal_count}, Parkinson: {test_parkinson_count})")
    """
    # 打印训练集和测试集的样本路径
    print("📂 训练集样本路径:")
    for i, idx in enumerate(train_ids):
        print(f"   [{i+1}] {seq_dir[idx][0]} (标签: {label[idx]})")
        
    print("📂 测试集样本路径:")
    for i, idx in enumerate(test_ids):
        print(f"   [{i+1}] {seq_dir[idx][0]} (标签: {label[idx]})")
    """
    train_source = DataSet(
        [seq_dir[i] for i in train_ids],
        [label[i] for i in train_ids],
        [seq_type[i] for i in train_ids],
        [view[i] for i in train_ids],
        cache, resolution, data_type, direction)

    test_source = DataSet(
        [seq_dir[i] for i in test_ids],
        [label[i] for i in test_ids],
        [seq_type[i] for i in test_ids],
        [view[i] for i in test_ids],
        cache, resolution, data_type, direction)

    return train_source, test_source