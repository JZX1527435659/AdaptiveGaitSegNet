#python test_BEST.py --ckpt_dir "C:\Users\User\Desktop\GaitSet-master\best_checkpoint" --batch_size 1 --cache=False

import os
import re
import sys
import io
import argparse
import subprocess
from datetime import datetime

def enable_utf8():
    """强制父进程 stdout/stderr 使用 UTF-8，避免 GBK 下抛错"""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def parse_args():
    p = argparse.ArgumentParser('Batch test all checkpoints')
    p.add_argument('--ckpt_dir',  required=True, help='*.ptm 文件存放目录')
    p.add_argument('--batch_size',default=1,  type=int, help='传给 test.py 的 --batch_size')
    p.add_argument('--cache',     default='False', choices=['True','False'], help='传给 test.py 的 --cache')
    p.add_argument('--python_cmd',default='python', help='Python 可执行命令')
    return p.parse_args()

def collect_iters(dir_path):
    """匹配所有 *-encoder.ptm，提取 iter 并排序"""
    pat = re.compile(r'.+-(\d+)-encoder\.ptm$', re.IGNORECASE)
    return sorted({pat.match(f).group(1) for f in os.listdir(dir_path) if pat.match(f)}, key=int)

def main():
    enable_utf8()
    args = parse_args()
    iters = collect_iters(args.ckpt_dir)
    if not iters:
        print('❌ 未检测到任何 encoder.ptm，退出。')
        return

    for it in iters:
        prefix = f'GaitSet_OUMVLP_5153_False_256_0.2_512_full_30-{it}'
        enc = prefix + '-encoder.ptm'
        enc_path = os.path.join(args.ckpt_dir, enc)
        if not os.path.isfile(enc_path):
            print(f'⚠ 跳过 iter={it}：缺少 `{enc}`')
            continue

        logp = os.path.join(args.ckpt_dir, f'{it}.log')
        print(f'\n=== 测试 iter={it}，日志 -> {logp} ===')
        with open(logp, 'w', encoding='utf-8') as lf:
            lf.write(f'=== 开始 iter={it} @ {datetime.now()} ===\n')

            cmd = [
                args.python_cmd,
                '-X', 'utf8',         # 全局 UTF-8 模式
                'test.py',
                '--iter',      it,
                '--batch_size',str(args.batch_size),
                '--cache',     args.cache,
                '--ckpt_dir',  args.ckpt_dir
            ]
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            proc = subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )

            for L in proc.stdout:
                print(L, end='')
                lf.write(L)
            proc.wait()

            msg = f'=== 结束 iter={it} code={proc.returncode} @ {datetime.now()} ===\n'
            lf.write(msg)
            print(msg)

    print('\n🎉 全部测试完毕')

if __name__=='__main__':
    main()
