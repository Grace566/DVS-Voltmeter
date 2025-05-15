import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image

# 定义源和目标目录
seq = 'CoreView_386'        # 指定序列
source_path = os.path.join('F:\\Dataset\\ZJU-MoCap_pre', seq, '1')
target_path = os.path.join('F:\\Dataset\\ZJU-MoCap_pre', seq, 'Src_Img\\1')
# temp_0 = np.load('G:\\Dataset\\3DPW\\3DPW_vid2e\\events\\courtyard_arguing_00\\0000000000.npz', allow_pickle=True)
# temp_7283 = np.load('G:\\Dataset\\3DPW\\3DPW_vid2e\\events\\courtyard_arguing_00\\0000007283.npz', allow_pickle=True)
# data = np.loadtxt('E:\\DVS-SIM\\src\\events\\courtyard_arguing_00.txt')
# print('test')

def process_sequence(sequence_path, target_sequence_path):
    # 确保目标目录存在
    os.makedirs(target_sequence_path, exist_ok=True)

    # 复制图像文件
    img_target_path = os.path.join(target_sequence_path)
    if not os.path.exists(img_target_path):
        os.makedirs(img_target_path)

    img_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.jpg')])        # jpg是图像，png是mask
    for img_file in img_files:
        # 假设 sequence_path 是源文件路径，img_target_path 是目标路径
        source_file = os.path.join(sequence_path, img_file)
        target_file = os.path.join(img_target_path, os.path.splitext(img_file)[0] + '.png')  # 更改扩展名为 .png

        # 检查源文件是否为 .jpg 格式
        if img_file.lower().endswith('.jpg'):
            # 使用 Pillow 读取 .jpg 文件并保存为 .png
            with Image.open(source_file) as img:
                img.save(target_file, format='PNG')
        else:
            # 如果不是 .jpg 文件，直接复制
            shutil.copy(source_file, target_file)

        # shutil.copy(os.path.join(sequence_path, img_file), os.path.join(img_target_path, img_file))

    # # 读取 fps.txt 文件并计算时间戳
    # fps_file_path = os.path.join(sequence_path, 'fps.txt')
    # with open(fps_file_path, 'r') as fps_file:
    #     fps = float(fps_file.read().strip())

    fps = 25

    timestamps = []
    for i in range(len(img_files)):
        timestamp = int((i / fps) * 1e6)  # 转换为微秒
        timestamps.append(timestamp)

    # 生成 info.txt 文件
    info_file_path = os.path.join(target_sequence_path, 'info.txt')
    with open(info_file_path, 'w') as info_file:
        for img_file, timestamp in zip(img_files, timestamps):
            info_file.write(f"{target_path}/{os.path.splitext(img_file)[0] + '.png'} {timestamp:012d}\n")


if os.path.isdir(source_path):
    process_sequence(source_path, target_path)
