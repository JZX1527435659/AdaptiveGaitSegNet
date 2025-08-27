import os
import shutil

# 定义源目录和目标目录
source_dir = r"C:\Users\User\Desktop\移动备份\Oppenheimer\万恶的资本主义市场经济\步态识别\OpenGait-master\output"
target_dir = r"C:\Users\User\Desktop\GaitSet-master\best_checkpoint"

# 确保目标目录存在，如果不存在则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历源目录并查找所有.pt文件
def copy_pt_files(src, dst):
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith(".pt"):  # 检查文件扩展名是否为.pt
                src_file = os.path.join(root, file)  # 获取文件的完整路径
                dst_file = os.path.join(dst, file)  # 目标路径
                shutil.copy2(src_file, dst_file)  # 复制文件
                print(f"复制文件: {src_file} -> {dst_file}")

# 调用函数执行复制操作
copy_pt_files(source_dir, target_dir)
print("所有.pt文件已复制完成！")