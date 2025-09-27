import os
import cv2
import numpy as np
from tqdm import tqdm


def generate_masks(silhouette):
    """生成边缘掩码（M_e）、内部掩码（M_i）和膨胀掩码（M_d）"""
    _, binary_mask = cv2.threshold(silhouette, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 内部掩码：对原始二值掩码腐蚀
    M_i = cv2.erode(binary_mask, kernel, iterations=1)
    # 膨胀掩码（M_d）：对原始二值掩码膨胀（原代码中的dilated）
    M_d = cv2.dilate(binary_mask, kernel, iterations=1)  # 重命名为M_d，明确是膨胀掩码
    # 边缘掩码：膨胀掩码与内部掩码的异或（边缘=膨胀区域-内部区域）
    M_e = cv2.bitwise_xor(M_d, M_i)

    # 归一化到0-1范围
    M_i = M_i / 255.0
    M_e = M_e / 255.0
    M_d = M_d / 255.0  # 同时归一化M_d
    return M_e, M_i, M_d  # 返回新增的M_d


def gait_synthesis(P, M_e, M_i):
    """合成最终剪影"""
    M_s = M_e * P + M_i
    M_s = np.clip(M_s, 0, 1)
    return M_s


def save_visualization(silhouette, M_e, M_i, M_d, M_s, visual_output_path):
    """保存每一步的可视化结果（新增M_d的保存）"""
    os.makedirs(visual_output_path, exist_ok=True)

    # 保存原始轮廓图
    cv2.imwrite(os.path.join(visual_output_path, "original_silhouette.png"), silhouette)

    # 保存膨胀掩码M_d（新增）
    M_d_visual = (M_d * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(visual_output_path, "M_d.png"), M_d_visual)

    # 保存内部掩码M_i
    M_i_visual = (M_i * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(visual_output_path, "M_i.png"), M_i_visual)

    # 保存边缘掩码M_e
    M_e_visual = (M_e * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(visual_output_path, "M_e.png"), M_e_visual)

    # 保存合成后的剪影M_s
    M_s_visual = (M_s * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(visual_output_path, "M_s.png"), M_s_visual)


def process_single_frame(silhouette_path, output_path, P, visual_output_root):
    """处理单张轮廓图并保存结果（适配M_d的传递）"""
    # 读取轮廓图
    silhouette = cv2.imread(silhouette_path, cv2.IMREAD_GRAYSCALE)
    if silhouette is None:
        return False  # 跳过损坏的图像

    # 生成掩码（新增M_d的接收）
    M_e, M_i, M_d = generate_masks(silhouette)  # 接收M_d
    M_s = gait_synthesis(P, M_e, M_i)

    # 保存合成结果
    cv2.imwrite(output_path, (M_s * 255).astype(np.uint8))

    # 构建可视化保存路径
    relative_path = os.path.relpath(output_path, start=OUTPUT_ROOT)
    visual_output_path = os.path.join(visual_output_root, relative_path)
    os.makedirs(os.path.dirname(visual_output_path), exist_ok=True)

    # 保存可视化中间结果（传入M_d）
    save_visualization(silhouette, M_e, M_i, M_d, M_s, os.path.dirname(visual_output_path))

    return True


def batch_process_dataset(input_root, output_root, visual_output_root):
    """批量处理数据集（逻辑不变，适配修改后的函数）"""
    for _id in tqdm(os.listdir(input_root), desc="处理ID"):
        id_input = os.path.join(input_root, _id)
        if not os.path.isdir(id_input):
            continue

        id_output = os.path.join(output_root, _id)
        os.makedirs(id_output, exist_ok=True)

        id_visual_output = os.path.join(visual_output_root, _id)
        os.makedirs(id_visual_output, exist_ok=True)

        for _seq in os.listdir(id_input):
            seq_input = os.path.join(id_input, _seq)
            if not os.path.isdir(seq_input):
                continue

            seq_output = os.path.join(id_output, _seq)
            os.makedirs(seq_output, exist_ok=True)

            seq_visual_output = os.path.join(id_visual_output, _seq)
            os.makedirs(seq_visual_output, exist_ok=True)

            for _view in os.listdir(seq_input):
                view_input = os.path.join(seq_input, _view)
                if not os.path.isdir(view_input):
                    continue

                view_output = os.path.join(seq_output, _view)
                os.makedirs(view_output, exist_ok=True)

                view_visual_output = os.path.join(seq_visual_output, _view)
                os.makedirs(view_visual_output, exist_ok=True)

                for frame in os.listdir(view_input):
                    if not frame.lower().endswith(('.png', '.jpg')):
                        continue

                    frame_input = os.path.join(view_input, frame)
                    frame_output = os.path.join(view_output, frame)

                    # 读取原始剪影并生成P
                    silhouette = cv2.imread(frame_input, cv2.IMREAD_GRAYSCALE)
                    if silhouette is None:
                        continue
                    P = (silhouette / 255.0).astype(np.float32)

                    # 处理并保存
                    process_single_frame(frame_input, frame_output, P, visual_output_root)


if __name__ == "__main__":
    # 配置输入输出路径
    INPUT_ROOT = r"C:\Users\User\Desktop\GaitSet-master\GAIT-IST dataset\diplegic"
    OUTPUT_ROOT = r"C:\Users\User\Desktop\GaitSet-master\output4\diplegic\output_synthesis"
    VISUAL_OUTPUT_ROOT = r"C:\Users\User\Desktop\GaitSet-master\output4\diplegic\visualization"

    # 批量处理
    batch_process_dataset(INPUT_ROOT, OUTPUT_ROOT, VISUAL_OUTPUT_ROOT)
    print("批量处理完成！")