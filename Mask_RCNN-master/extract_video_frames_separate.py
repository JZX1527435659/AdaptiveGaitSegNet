import cv2
import numpy as np
import os
from mrcnn.config import Config
import tensorflow as tf

from mrcnn import model as modellib
from mrcnn import utils

# 忽略TensorFlow警告
import warnings

warnings.filterwarnings("ignore")


# 1. 模型配置
class InferenceConfig(Config):
    NAME = "coco_preserve_size"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO类别

    # 检测尺寸配置
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    IMAGE_SHAPE = (640, 640, 3)
    IMAGE_RESIZE_MODE = "square"

    # 检测参数
    DETECTION_MAX_INSTANCES = 5
    DETECTION_MIN_CONFIDENCE = 0.6
    DETECTION_NMS_THRESHOLD = 0.3


config = InferenceConfig()
print("模型配置：")
print(f"检测尺寸：{config.IMAGE_MIN_DIM}~{config.IMAGE_MAX_DIM} | 输出尺寸：保持原始")

# 2. 模型加载
MODEL_DIR = "./model_cache"
COCO_MODEL_PATH = r"C:\Users\User\Desktop\wyx-GaitSet-master\Mask_RCNN-master\mask_rcnn_coco.h5"

if not os.path.exists(COCO_MODEL_PATH):
    raise FileNotFoundError(
        f"权重文件不存在：{COCO_MODEL_PATH}\n"
        "请从 https://github.com/matterport/Mask_RCNN/releases 下载"
    )
os.makedirs(MODEL_DIR, exist_ok=True)

model = modellib.MaskRCNN(
    mode="inference",
    model_dir=MODEL_DIR,
    config=config
)

try:
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    print("权重加载成功")
except Exception as e:
    print(f"权重加载失败：{str(e)}")


# 3. 人体轮廓提取（核心：简化尺寸映射计算）
def extract_person_contour(image):
    try:
        original_height, original_width = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 手动计算缩放比例（完全绕过window参数）
        # 1. 计算原始图像的缩放因子
        scale = min(
            config.IMAGE_MAX_DIM / original_width,
            config.IMAGE_MAX_DIM / original_height
        )
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # 2. 手动缩放图像（保持比例）
        resized_image = cv2.resize(
            rgb_image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # 3. 计算填充尺寸（使图像达到模型要求的尺寸）
        pad_w = (config.IMAGE_MAX_DIM - new_width) // 2
        pad_h = (config.IMAGE_MAX_DIM - new_height) // 2
        resized_image = cv2.copyMakeBorder(
            resized_image,
            pad_h, config.IMAGE_MAX_DIM - new_height - pad_h,
            pad_w, config.IMAGE_MAX_DIM - new_width - pad_w,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # 模型推理
        results = model.detect([resized_image], verbose=0)
        r = results[0]

        contour_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
        COCO_PERSON_ID = 1
        MIN_CONFIDENCE = 0.5

        if len(r["class_ids"]) == 0:
            return contour_image

        for i in range(len(r["class_ids"])):
            if r["class_ids"][i] == COCO_PERSON_ID and r["scores"][i] > MIN_CONFIDENCE:
                mask = r["masks"][:, :, i].astype(np.uint8)
                if not np.any(mask):
                    continue

                # 1. 移除填充区域
                mask_cropped = mask[pad_h:pad_h + new_height, pad_w:pad_w + new_width]

                # 2. 还原到原始图像尺寸
                mask_resized = cv2.resize(
                    mask_cropped,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST
                )

                # 3. 填充轮廓
                contour_image[mask_resized > 0] = [255, 255, 255]

        return contour_image

    except Exception as e:
        print(f"帧处理错误：{str(e)}")
        return np.zeros_like(image)


# 4. 视频帧处理
def extract_video_frames(video_path, output_root):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在：{video_path}")

    orig_dir = os.path.join(output_root, "original")
    contour_dir = os.path.join(output_root, "contour")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(contour_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频原始尺寸：{original_width}x{original_height} | {fps:.1f}fps | 共{total_frames}帧")
    print(f"将提取全部帧 | 预计{total_frames}组")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读取完毕退出循环

        contour_frame = extract_person_contour(frame)

        # 保存帧（尺寸严格一致）
        orig_path = os.path.join(orig_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(orig_path, frame)

        contour_path = os.path.join(contour_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(contour_path, contour_frame)

        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"进度：{progress:.1f}% | 已保存{frame_count}组", end='\r')

    cap.release()
    print(f"\n处理完成！")
    print(f"原始帧：{os.path.abspath(orig_dir)}")
    print(f"轮廓帧：{os.path.abspath(contour_dir)}")


# 5. 主函数
if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\User\Desktop\Mask_RCNN-master\video.mp4"
    extract_video_frames(VIDEO_PATH, output_root="video_output")