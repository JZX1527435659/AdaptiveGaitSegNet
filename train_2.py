from modelfile.initialization_2 import initialization
from config_2 import conf
import argparse
import torch
import os


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train (GAIT-IST silhouettes, binary classification)')
    parser.add_argument('--cache', default=True, type=boolean_string,
                        help='是否将训练数据一次性缓存到内存中，默认 True')
    opt = parser.parse_args()

    # 初始化模型与数据（已适配 GAIT-IST）
    model, _ = initialization(conf, train=opt.cache)

    print("\nTraining START (front/back 需在 config_1.py 中切换 direction)…\n")
    model.fit()
    print("\n🏁 Training COMPLETE")
