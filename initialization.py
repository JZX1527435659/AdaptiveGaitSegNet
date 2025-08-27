import os
from copy import deepcopy

import numpy as np

from .utils import load_data
from .model import Model


def initialize_data(config, train=False, test=False):
    """
    初始化训练和测试数据源
    :param config: 配置文件
    :param train: 是否加载训练数据
    :param test: 是否加载测试数据
    :return: train_source, test_source
    """
    print("Initializing data source...")
    # 是否缓存全部数据：若训练或测试为 True 则启用缓存机制
    train_source, test_source = load_data(
        **config['data'],
        cache=(train or test)
    )

    if train:
        print("Loading training data into memory...")
        train_source.load_all_data()
    if test:
        print("Loading test data into memory...")
        test_source.load_all_data()

    print("✅ Data initialization complete.")
    return train_source, test_source


def initialize_model(config, train_source, test_source):
    """
    初始化模型并封装参数
    :param config: 配置文件
    :param train_source: 训练集对象
    :param test_source: 测试集对象
    :return: 模型实例, 模型保存名
    """
    print("Initializing model...")
    data_config = config['data']
    model_config = config['modelfile']
    model_param = deepcopy(model_config)

    # 填充训练与测试数据源
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_config['pid_num']

    # 构造保存名（用于模型保存目录）
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str, [
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    # 初始化模型对象
    m = Model(**model_param)
    print("✅ Model initialization complete.")
    return m, model_param['save_name']


def initialization(config, train=False, test=False):
    """
    主初始化函数：切换工作目录，设定 CUDA 环境，初始化数据与模型
    :param config: 配置文件
    :param train: 是否加载训练数据
    :param test: 是否加载测试数据
    :return: 模型实例, 模型保存名
    """
    print("🔧 Initializing...")
    WORK_PATH = config['WORK_PATH']
    os.chdir(WORK_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]

    # 初始化数据
    train_source, test_source = initialize_data(config, train, test)

    # 初始化模型
    return initialize_model(config, train_source, test_source)
