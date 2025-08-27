conf = {
    "WORK_PATH": r"C:\Users\User\Desktop\zwc-GaitSet-master\work",
    "CUDA_VISIBLE_DEVICES": "0",

    "data": {
        'dataset_path': r"C:\Users\User\Desktop\wyx-GaitSet-master\GaitDatasetA-silh\output_synthesis_rotate",
        'resolution': '64',             # 图片分辨率，默认即可
        'dataset': 'CASIA-B',
        'pid_num': 12,                  # 每个 batch 中的不同行人数量，建议设小一点
        'pid_shuffle': False,           # 是否打乱 pid，设为 False 更稳定
    },

    "modelfile": {
        'hidden_dim': 128,  # 特征维度调小（减少显存占用）
        'lr': 1e-4,
        'hard_or_full_trip': 'full',  # 可尝试 'hard'（减少负样本计算）进一步节省显存
        'batch_size': (4, 2),  # 总 batch size 为 8
        'restore_iter': 0,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 0,  # Windows 环境建议设为 0，避免多进程报错
        'frame_num': 20,  # 一条 gait 序列的帧数，调小减显存
        'model_name': 'GaitSet',
    },
}
