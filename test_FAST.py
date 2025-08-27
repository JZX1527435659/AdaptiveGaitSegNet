#显式指定 checkpoint（比如 4000）：python test.py --iter 4000
#python test_FAST.py --cache=False

import os
import numpy as np
import argparse
from datetime import datetime

from modelfile.initialization import initialization
from modelfile.utils import evaluation
from config import conf  # 默认从 config/__init__.py 获取 conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Fast Test for GaitSet')
parser.add_argument('--iter', default=80000, type=int,
                    help='Iteration of checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='If TRUE, loads test data into memory for faster testing')
opt = parser.parse_args()


def de_diag(acc, each_angle=False):
    """Exclude identical-view cases in evaluation."""
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    return result if each_angle else np.mean(result)


# === 自动补充缺失字段 ===
if 'output_dir' not in conf:
    conf['output_dir'] = './output'
if 'experiment_name' not in conf:
    conf['experiment_name'] = 'default_exp'
if 'data' not in conf:
    conf['data'] = 'CASIA-B'

# === 初始化模型 ===
print('🔧 Initializing...')
m = initialization(conf, test=opt.cache)[0]

# === 加载权重 ===
print(f'📦 Loading the model checkpoint from iteration {opt.iter}...')
m.load(opt.iter)

# === 开始测试 ===
print('🚀 Transforming (inference on test set)...')
start_time = datetime.now()
test = m.transform('test', opt.batch_size)

print('📊 Evaluating...')
acc = evaluation(test, conf['data'])
print('✅ Evaluation complete. Time cost:', datetime.now() - start_time)

# === 打印指标 ===
np.set_printoptions(precision=2, floatmode='fixed')
for i in range(1):  # 只输出 Rank-1
    print(f'\n=== Rank-{i+1} (Include identical-view cases) ===')
    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        np.mean(acc[0, :, :, i]),
        np.mean(acc[1, :, :, i]),
        np.mean(acc[2, :, :, i])))

    print(f'\n=== Rank-{i+1} (Exclude identical-view cases) ===')
    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        de_diag(acc[0, :, :, i]),
        de_diag(acc[1, :, :, i]),
        de_diag(acc[2, :, :, i])))

    print(f'\n=== Rank-{i+1} of each angle (Exclude identical-view cases) ===')
    print('NM:', de_diag(acc[0, :, :, i], True))
    print('BG:', de_diag(acc[1, :, :, i], True))
    print('CL:', de_diag(acc[2, :, :, i], True))