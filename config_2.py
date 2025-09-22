conf = {
    "WORK_PATH": r"C:\\Users\\User\\Desktop\\zwc-GaitSet-master\\work",
    "CUDA_VISIBLE_DEVICES": "0",

    # 数据相关配置（适配 GAIT-IST 预处理后的目录结构 gaitist_output）
    "data": {
        'dataset_path': r"C:\\Users\\User\\Desktop\\zwc-GaitSet-master\\gaitist_output",
        'resolution': '64',             # 输入分辨率（保持与原网络一致）
        'dataset': 'GAIT-IST',          # 仅用于保存与日志标识
        'pid_num': 2,                   # 二分类：正常(0)/帕金森(1)
        'pid_shuffle': False,           # 为保证复现，默认不打乱
        'data_type': 'silhouettes',     # 使用silhouettes/能量图
        'direction': 'back',           # 方向，可在测试脚本中切换为 back
        'balance': False,               # True=类均衡采样; False=使用全部样本
    },

    # 训练相关配置（不改网络结构，仅保持原始 SetNet 超参格式）
    "modelfile": {
        'hidden_dim': 128,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (4, 2),           # P x M（与原实现一致）
        'restore_iter': 0,
        'total_iter': 10000,
        'margin': 0.2,
        'num_workers': 0,               # Windows 建议为 0
        'frame_num': 20,
        'model_name': 'GaitSet',
        
        # 早停相关参数
        'early_stop_patience': 50,      # 容忍多少次验证损失没有改善
        'early_stop_delta': 1e-4,       # 最小改善阈值
    },
}


       