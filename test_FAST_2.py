import argparse
from datetime import datetime
import numpy as np

from modelfile.initialization_2 import initialization
from modelfile.utils.evaluator_2 import evaluation, evaluate_binary_classification, save_confusion_matrix
from config_2 import conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Fast Test for GAIT-IST (silhouettes)')
parser.add_argument('--iter', default=200, type=int,
                    help='加载的 checkpoint 迭代数，默认 200')
parser.add_argument('--batch_size', default=1, type=int,
                    help='测试 batch size，默认 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='是否将测试数据缓存到内存中，默认 False')
opt = parser.parse_args()


def run_once(direction: str):
    """按指定方向(front/back)测试一次，返回总体准确率(二分类)。"""
    conf['data']['direction'] = direction
    print(f"\n🔧 Initializing for direction = {direction} …")
    m, save_name = initialization(conf, test=opt.cache)

    print(f'📦 Loading the model checkpoint from iteration {opt.iter}...')
    data_type = conf['data'].get('data_type', 'silhouettes')
    m.load(opt.iter, data_type=data_type, direction=direction)

    print('🚀 Transforming (inference on test set)…')
    start_time = datetime.now()
    test = m.transform('test', opt.batch_size)
    print('✅ Transform complete. Time cost:', datetime.now() - start_time)

    # 分类评测（二分类）
    print('📊 Classifying…')
    test_logits = m.transform_logits('test', opt.batch_size)
    acc, precision, recall, f1, cm = evaluate_binary_classification(test_logits)
    print(f'   Acc={acc:.2f}%  P={precision:.2f}%  R={recall:.2f}%  F1={f1:.2f}%')

    # 保存混淆矩阵，使用confusion_data_type_direction的形式
    save_path = f'confusion_{data_type}_{direction}.png'
    save_confusion_matrix(cm, save_path)
    print(f'   Confusion matrix saved to {save_path}')

    overall = acc
    return overall


if __name__ == '__main__':
    # 只测试配置文件中指定的方向
    direction = conf['data']['direction']
    acc = run_once(direction)
    
    print("\n===== FuseLGNet (silhouettes) =====")
    # 注意：overall 已经是百分数(0-100)，不再二次乘以100
    print(f"{direction}\t{acc:.2f}%")
