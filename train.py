from modelfile.initialization import initialization
from config import conf
import argparse
import torch

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--cache', default=True, type=boolean_string,
                        help='cache: if set as TRUE all the training data will be loaded at once '
                             'before the training start. Default: TRUE')
    opt = parser.parse_args()

    print("🔧 Initializing...")
    # 初始化模型和数据（根据返回值严格保留两个）
    model, loader = initialization(conf, train=opt.cache)
    print("✅ Initialization complete.\n")


    print("🧠 Model architecture:")
    # 模型结构不打印了，避免 AttributeError
    # print(model.net)  # ❌ 这一行会报错，故注释
    print("\nTraining START\n")

    # 开始训练
    model.fit()
    print("\n🏁 Training COMPLETE")
