# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from warnings import warn
from time import sleep
import argparse
import math
from multiprocessing import Pool, TimeoutError as MP_TimeoutError

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


wd = os.getcwd()
input_path = os.path.join(wd, r'C:\Users\User\Desktop\GaitSet-master\output1\output_synthesis')
output_path = os.path.join(wd, r'C:\Users\User\Desktop\GaitSet-master\output1\output_synthesis_rotate')

parser = argparse.ArgumentParser(description='剪影形态特征校正方向')
parser.add_argument('--input_path', default=input_path, type=str)
parser.add_argument('--output_path', default=output_path, type=str)
parser.add_argument('--log_file', default='./pretreatment.log', type=str)
parser.add_argument('--log', default=False, type=boolean_string)
parser.add_argument('--worker_num', default=1, type=int)
parser.add_argument('--debug', default=False, type=boolean_string)
opt = parser.parse_args()

INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num
DEBUG = opt.debug
T_H, T_W = 64, 64  # 目标尺寸


def log2str(pid, comment, logs):
    str_log = ''
    if isinstance(logs, str):
        logs = [logs]
    for log in logs:
        str_log += f"# JOB {pid} : --{comment}-- {log}\n"
    return str_log


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as f:
            f.write(str_log)
    if comment in [START, FINISH] and pid % 500 == 0:
        print(str_log, end='')


def get_center_and_axes(img):
    y_coords, x_coords = np.where(img > 0)
    if len(y_coords) < 10:
        return None, None, None, None

    cx, cy = np.mean(x_coords), np.mean(y_coords)
    x, y = x_coords - cx, y_coords - cy
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi

    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    return cx, cy, angle, (max_y - min_y, max_x - min_x)


def rotate_image(image, angle, center):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def silhouette_direction_correction(img):
    """专为剪影设计的方向判断算法"""
    # 1. 计算每行的水平宽度（剪影左右边界距离）
    row_widths = []
    for y in range(img.shape[0]):
        x_coords = np.where(img[y, :] > 0)[0]
        if len(x_coords) < 2:
            row_widths.append(0)
        else:
            row_widths.append(np.max(x_coords) - np.min(x_coords))
    row_widths = np.array(row_widths)

    # 2. 分割上下区域（上30%为头部候选区，下40%为脚部候选区）
    head_region = row_widths[:int(len(row_widths) * 0.3)]
    foot_region = row_widths[-int(len(row_widths) * 0.4):]

    # 特征1：头部平均宽度 < 脚部平均宽度
    head_mean = np.mean(head_region[head_region > 0]) if np.any(head_region > 0) else 0
    foot_mean = np.mean(foot_region[foot_region > 0]) if np.any(foot_region > 0) else 0
    feature1 = head_mean < foot_mean * 0.85  # 头部更窄

    # 3. 检测肩部凸起（上半部分宽度突变）
    shoulder_idx = int(len(row_widths) * 0.2)  # 肩部大致位置
    shoulder_width = row_widths[shoulder_idx] if shoulder_idx < len(row_widths) else 0
    head_max = np.max(head_region) if np.any(head_region > 0) else 0
    feature2 = shoulder_width > head_max * 1.2  # 肩部比头部宽

    # 4. 检测脚部收敛（从下往上宽度递增）
    foot_trend = np.mean(np.diff(foot_region[foot_region > 0])) if len(foot_region) > 1 else 0
    feature3 = foot_trend > 0  # 脚部从下到上逐渐变宽

    # 综合3个特征判断方向（至少2个特征满足则为正向）
    positive_features = sum([feature1, feature2, feature3])
    if positive_features < 2:
        return np.flipud(img)  # 方向错误，翻转
    return img


def align_img(img, seq_info, frame_name, pid):
    if img.sum() <= 10000:
        log_print(pid, WARNING, f'seq:{seq_info}, frame:{frame_name}, 有效像素不足')
        return None

    cx, cy, angle, (height, width) = get_center_and_axes(img)
    if cx is None:
        log_print(pid, WARNING, f'seq:{seq_info}, frame:{frame_name}, 无法计算主轴')
        return None

    # 旋转校正倾斜
    rotated = rotate_image(img, -angle, (cx, cy))

    # 裁剪边界框
    y_coords, x_coords = np.where(rotated > 0)
    if not y_coords.size:
        return None
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    rotated_cropped = rotated[min_y:max_y + 1, min_x:max_x + 1]

    # 缩放与居中
    h, w = rotated_cropped.shape
    if h == 0:
        return None
    scale = T_H / h
    scaled_w = int(w * scale)
    scaled = cv2.resize(rotated_cropped, (scaled_w, T_H), interpolation=cv2.INTER_CUBIC)

    if scaled_w < T_W:
        pad_left = (T_W - scaled_w) // 2
        pad_right = T_W - scaled_w - pad_left
        aligned = np.pad(scaled, ((0, 0), (pad_left, pad_right)), mode='constant')
    else:
        start_x = (scaled_w - T_W) // 2
        aligned = scaled[:, start_x:start_x + T_W]

    # 核心：剪影形态特征校正方向
    aligned = silhouette_direction_correction(aligned)

    # 调试保存
    if DEBUG:
        debug_dir = os.path.join(OUTPUT_PATH, 'debug', *seq_info)
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f'final_{frame_name}'), aligned)

    return aligned.astype('uint8')


def cut_pickle(seq_info, pid):
    seq_name = '-'.join(seq_info)
    log_print(pid, START, seq_name)
    seq_path = os.path.join(INPUT_PATH, *seq_info)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    os.makedirs(out_dir, exist_ok=True)

    frame_list = sorted(os.listdir(seq_path))
    count_frame = 0
    for frame_name in frame_list:
        if not frame_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        frame_path = os.path.join(seq_path, frame_name)
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        aligned = align_img(img, seq_info, frame_name, pid)
        if aligned is not None:
            cv2.imwrite(os.path.join(out_dir, frame_name), aligned)
            count_frame += 1

    if count_frame < 5:
        log_print(pid, WARNING, f'seq:{seq_name}, 有效帧不足5帧')
    log_print(pid, FINISH, f'保存{count_frame}帧至{out_dir}')
    return count_frame


if __name__ == '__main__':
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    print(f'预处理开始（剪影形态特征校正）\n输入路径: {INPUT_PATH}\n输出路径: {OUTPUT_PATH}')

    pool = Pool(WORKERS)
    results = []
    pid = 0

    id_list = sorted(os.listdir(INPUT_PATH))
    for _id in id_list:
        id_path = os.path.join(INPUT_PATH, _id)
        if not os.path.isdir(id_path):
            continue
        for _seq_type in sorted(os.listdir(id_path)):
            seq_type_path = os.path.join(id_path, _seq_type)
            if not os.path.isdir(seq_type_path):
                continue
            for _view in sorted(os.listdir(seq_type_path)):
                view_path = os.path.join(seq_type_path, _view)
                if not os.path.isdir(view_path):
                    continue
                seq_info = [_id, _seq_type, _view]
                results.append(pool.apply_async(cut_pickle, (seq_info, pid)))
                sleep(0.02)
                pid += 1

    pool.close()
    unfinish = 1
    while unfinish > 0:
        unfinish = 0
        for i, res in enumerate(results):
            try:
                res.get(timeout=0.1)
            except MP_TimeoutError:
                unfinish += 1
            except Exception as e:
                print(f'\n错误: PID {i}, 类型 {type(e)}')
                raise e
    pool.join()