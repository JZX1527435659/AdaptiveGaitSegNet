import os
import cv2
import numpy as np


def preprocess_frame(frame):
    """图像预处理步骤：保留轮廓内部+增强边缘"""
    # 1. 灰度化处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用CLAHE算法增强局部对比度，避免内部暗区被误判
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 提高clipLimit增强对比度
    # gray_enhanced = clahe.apply(gray)

    # 2. 二值化处理（保留完整前景区域，包括内部）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Sobel边缘检测（仅用于增强轮廓边缘）
    sobelx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))
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

    # 解析视频路径，构建对应的输出目录结构
    # 例如：pretreatment\001\bg-01\000\video.mp4 -> video/output/001/bg-01/000/video/
    video_dir = os.path.dirname(video_path)
    relative_path = os.path.relpath(video_dir, "pretreatment")  # 获取相对于pretreatment的路径
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    # 构建输出目录（保持与源目录相同的层级结构）
    output_dir = os.path.join(output_root, relative_path, file_name)
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

        # 保存中间帧
        if start_idx <= frame_count <= end_idx:
            save_path = os.path.join(output_dir, f"{frame_count}.png")
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
    # 视频根目录（pretreatment及其子目录）
    video_root = r"C:\Users\User\Desktop\wyx-GaitSet-master\output\video"
    # 输出根目录
    output_root = r"C:\Users\User\Desktop\wyx-GaitSet-master\output1\pretreatment1"

    # 查找所有视频文件
    all_videos = find_all_videos(video_root)

    if not all_videos:
        print(f"警告：在 {video_root} 目录下未找到任何视频文件")
    else:
        print(f"找到 {len(all_videos)} 个视频文件，开始处理...")
        for video_path in all_videos:
            process_video(video_path, output_root)
        print("所有视频处理完成！")
