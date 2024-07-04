import os
import shutil

# 定义源和目标目录
source_root = r'I:\Dataset\3DPW\3DPW_vid2e\imageFiles'
target_root = r'I:\Dataset\3DPW\3DPW_vid2e\image_DVSVoltmeter'


def process_sequence(sequence_path, target_sequence_path):
    # 确保目标目录存在
    os.makedirs(target_sequence_path, exist_ok=True)

    # 复制图像文件
    img_source_path = os.path.join(sequence_path, 'imgs')
    img_target_path = os.path.join(target_sequence_path)
    if not os.path.exists(img_target_path):
        os.makedirs(img_target_path)

    img_files = sorted([f for f in os.listdir(img_source_path) if f.endswith('.png')])
    for img_file in img_files:
        shutil.copy(os.path.join(img_source_path, img_file), os.path.join(img_target_path, img_file))

    # 读取 fps.txt 文件并计算时间戳
    fps_file_path = os.path.join(sequence_path, 'fps.txt')
    with open(fps_file_path, 'r') as fps_file:
        fps = float(fps_file.read().strip())

    timestamps = []
    for i in range(len(img_files)):
        timestamp = int((i / fps) * 1e6)  # 转换为微秒
        timestamps.append(timestamp)

    # 生成 info.txt 文件
    info_file_path = os.path.join(target_sequence_path, 'info.txt')
    with open(info_file_path, 'w') as info_file:
        for img_file, timestamp in zip(img_files, timestamps):
            info_file.write(f'./data_samples/interp/{os.path.basename(sequence_path)}/{img_file} {timestamp:012d}\n')


# 遍历所有序列文件夹并处理
for sequence_folder in os.listdir(source_root):
    sequence_path = os.path.join(source_root, sequence_folder)
    target_sequence_path = os.path.join(target_root, sequence_folder)
    if os.path.isdir(sequence_path):
        process_sequence(sequence_path, target_sequence_path)
