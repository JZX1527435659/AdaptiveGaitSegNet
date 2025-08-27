# python test_ALL.py --cache=True --batch_size=16

import os
import numpy as np
import argparse
from datetime import datetime
from modelfile.initialization import initialization
from modelfile.utils import evaluation
from config import conf

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

parser = argparse.ArgumentParser(description='Batch Test for GaitSet')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string, help='If TRUE, loads test data into memory')
opt = parser.parse_args()

# 设置日志文件路径
log_file_path = r'C:\Users\User\Desktop\rjh-GaitSet-master - Copy - Copy\test_log1.txt'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

def de_diag(acc, each_angle=False):
    """Exclude identical-view cases in evaluation."""
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    return result if each_angle else np.mean(result)

# 初始化模型
if 'output_dir' not in conf:
    conf['output_dir'] = './output'
if 'experiment_name' not in conf:
    conf['experiment_name'] = 'default_exp'
if 'data' not in conf:
    conf['data'] = 'CASIA-B'

with open(log_file_path, 'w', encoding='utf-8') as log_file:
    for iter_value in range(100, 80001, 100):
        log_file.write(f'\n=== Testing checkpoint at iter {iter_value} ===\n')
        print(f'\n=== Testing checkpoint at iter {iter_value} ===')

        log_file.write('🔧 Initializing model...\n')
        print('🔧 Initializing model...')
        m = initialization(conf, test=opt.cache)[0]

        log_file.write(f'📦 Loading model from iteration {iter_value}...\n')
        print(f'📦 Loading model from iteration {iter_value}...')
        m.load(iter_value)

        log_file.write('🚀 Running inference...\n')
        print('🚀 Running inference...')
        start_time = datetime.now()
        test = m.transform('test', opt.batch_size)

        log_file.write('📊 Evaluating results...\n')
        print('📊 Evaluating results...')
        acc = evaluation(test, conf['data'])
        elapsed = datetime.now() - start_time
        log_file.write(f'✅ Done. Time cost: {elapsed}\n')
        print(f'✅ Done. Time cost: {elapsed}')

        # 打印 Rank-1
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            log_file.write(f'\n=== Rank-{i+1} (Include identical-view cases) ===\n')
            print(f'\n=== Rank-{i+1} (Include identical-view cases) ===')
            log_file.write('NM: %.3f,\tBG: %.3f,\tCL: %.3f\n' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))
            print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))

            log_file.write(f'\n=== Rank-{i+1} (Exclude identical-view cases) ===\n')
            print(f'\n=== Rank-{i+1} (Exclude identical-view cases) ===')
            log_file.write('NM: %.3f,\tBG: %.3f,\tCL: %.3f\n' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))
            print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))

            log_file.write(f'\n=== Rank-{i+1} of each angle (Exclude identical-view cases) ===\n')
            log_file.write('NM: ' + np.array2string(de_diag(acc[0, :, :, i], True)) + '\n')
            log_file.write('BG: ' + np.array2string(de_diag(acc[1, :, :, i], True)) + '\n')
            log_file.write('CL: ' + np.array2string(de_diag(acc[2, :, :, i], True)) + '\n')

            print(f'\n=== Rank-{i+1} of each angle (Exclude identical-view cases) ===')
            print('NM:', de_diag(acc[0, :, :, i], True))
            print('BG:', de_diag(acc[1, :, :, i], True))
            print('CL:', de_diag(acc[2, :, :, i], True))
