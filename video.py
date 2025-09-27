import os
import cv2
import numpy as np


def preprocess_frame(frame):
    """图像预处理步骤：保留轮廓内部+增强边缘"""
    # 1. 灰度化处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 二值化处理（保留完整前景区域，包括内部）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Sobel边缘检测（仅用于增强轮廓边缘）
    sobelx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobelx))
    sobel_edges = cv2.bitwise_or(sobelx, sobely)
    _, sobel_edges = cv2.threshold(sobel_edges, 50, 255, cv2.THRESH_BINARY)

    # 4. 融合：保留原始二值化的内部区域 + 叠加Sobel增强的边缘
    merged = cv2.bitwise_or(binary, sobel_edges)

    # 5. 滤波去噪
    denoised = cv2.medianBlur(merged, 3)

    # 6. 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel, iterations=1)  # 去噪
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=2)  # 填充

    return denoised


def process_video(video_path, output_root):
    print(f"处理视频: {video_path}")

    # ====================== 解析路径层级 ======================
    # 路径示例: Dataset/001/bg-01/001-bg-01-000.avi
    # 拆分后: ['001', 'bg-01', '001-bg-01-000.avi']（相对于 Dataset 目录）
    relative_path = os.path.relpath(video_path, start=video_root)
    parts = relative_path.split(os.sep)

    # 校验层级（必须是: ID -> 类型目录 -> 视频文件）
    if len(parts) != 3:
        print(f"路径 {video_path} 层级错误，跳过！相对路径拆分后为 {parts}")
        return

    id_part, type_dir, video_file = parts
    # 提取视频序号（如 000，从视频文件名中拆分）
    # 假设视频文件名格式为: 001-bg-01-000.avi → 拆分出 000
    video_seq = video_file.split('-')[-1].split('.')[0]  # 获取最后一段数字

    # ====================== 构建输出目录 ======================
    # 输出结构: output_root/ID/类型目录/视频序号/
    # 示例: pretreatment/001/bg-01/000/
    output_dir = os.path.join(output_root, id_part, type_dir, video_seq)
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_idx = int(total_frames * 0.3)
    end_idx = int(total_frames * 0.9)

    # 创建背景消除器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=1000,
        detectShadows=False
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理
        preprocessed = preprocess_frame(frame)

        # 背景消除
        fg_mask = bg_subtractor.apply(preprocessed)

        # 保存中间帧（命名格式: ID-类型目录-视频序号-帧序号.png）
        if start_idx <= frame_count <= end_idx:
            img_name = f"{id_part}-{type_dir}-{video_seq}-{str(frame_count).zfill(3)}.png"
            save_path = os.path.join(output_dir, img_name)
            cv2.imwrite(save_path, fg_mask)

        # 实时显示（按q退出）
        cv2.imshow('Processed Frame', fg_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"视频处理完成，保存 {end_idx - start_idx + 1} 帧到 {output_dir}")


def find_all_videos(root_dir):
    """递归查找root_dir下所有视频文件"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(video_extensions):
                video_path = os.path.join(dirpath, filename)
                video_paths.append(video_path)

    return video_paths


if __name__ == "__main__":
    # ====================== 配置路径 ======================
    # 数据集根目录（必须包含 Dataset/001/... 结构）
    video_root = r"C:\Users\User\Desktop\GaitSet-master\Dataset"
    # 输出根目录（生成 pretreatment 结构）
    output_root = r"C:\Users\User\Desktop\GaitSet-master\output2\pretreatment"

    # 查找所有视频文件
    all_videos = find_all_videos(video_root)

    if not all_videos:
        print(f"警告：在 {video_root} 目录下未找到任何视频文件")
    else:
        print(f"找到 {len(all_videos)} 个视频文件，开始处理...")
        for video_path in all_videos:
            process_video(video_path, output_root)
        print("所有视频处理完成！")