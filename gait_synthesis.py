import os
import cv2
import numpy as np
from tqdm import tqdm


def generate_masks(silhouette):
    """生成边缘掩码（M_e）和内部掩码（M_i），同之前的实现"""
    _, binary_mask = cv2.threshold(silhouette, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    M_i = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    M_e = cv2.bitwise_xor(dilated, M_i)
    M_i = M_i / 255.0
    M_e = M_e / 255.0
    return M_e, M_i


def gait_synthesis(P, M_e, M_i):
    """合成最终剪影，同之前的实现"""
    M_s = M_e * P + M_i
    M_s = np.clip(M_s, 0, 1)
    return M_s


def process_single_frame(silhouette_path, output_path, P):
    """处理单张轮廓图并保存结果"""
    # 读取轮廓图
    silhouette = cv2.imread(silhouette_path, cv2.IMREAD_GRAYSCALE)
    if silhouette is None:
        return False  # 跳过损坏的图像

    # 生成掩码并合成剪影
    M_e, M_i = generate_masks(silhouette)
    M_s = gait_synthesis(P, M_e, M_i)  # P为分割网络输出（此处用随机值模拟）

    # 保存结果
    cv2.imwrite(output_path, (M_s * 255).astype(np.uint8))
    return True


def batch_process_dataset(input_root, output_root):
    """
    批量处理三级结构数据集
    input_root: 输入根目录，结构为 id/seq/view/frame.png
    output_root: 输出根目录，保持相同的三级结构
    """
    # 遍历所有id
    for _id in tqdm(os.listdir(input_root), desc="处理ID"):
        id_input = os.path.join(input_root, _id)
        if not os.path.isdir(id_input):
            continue

        id_output = os.path.join(output_root, _id)
        os.makedirs(id_output, exist_ok=True)

        # 遍历每个id下的seq
        for _seq in os.listdir(id_input):
            seq_input = os.path.join(id_input, _seq)
            if not os.path.isdir(seq_input):
                continue

            seq_output = os.path.join(id_output, _seq)
            os.makedirs(seq_output, exist_ok=True)

            # 遍历每个seq下的view
            for _view in os.listdir(seq_input):
                view_input = os.path.join(seq_input, _view)
                if not os.path.isdir(view_input):
                    continue

                view_output = os.path.join(seq_output, _view)
                os.makedirs(view_output, exist_ok=True)

                # 处理当前view下的所有帧
                for frame in os.listdir(view_input):
                    if not frame.lower().endswith(('.png', '.jpg')):
                        continue

                    frame_input = os.path.join(view_input, frame)
                    frame_output = os.path.join(view_output, frame)

                    # 模拟分割网络输出P（实际应用中需替换为真实模型输出）
                    # 这里用轮廓图归一化后的值作为P的近似
                    silhouette = cv2.imread(frame_input, cv2.IMREAD_GRAYSCALE)
                    P = (silhouette / 255.0).astype(np.float32)

                    # 处理并保存
                    process_single_frame(frame_input, frame_output, P)


if __name__ == "__main__":
    # 配置输入输出路径
    INPUT_ROOT = r"C:\Users\User\Desktop\wyx-GaitSet-master\GaitDatasetA-silh\pretreatment"  # 输入目录
    OUTPUT_ROOT = r"C:\Users\User\Desktop\wyx-GaitSet-master\GaitDatasetA-silh\output_synthesis_7_7"  # 输出目录

    # 批量处理
    batch_process_dataset(INPUT_ROOT, OUTPUT_ROOT)
    print("批量处理完成！")
