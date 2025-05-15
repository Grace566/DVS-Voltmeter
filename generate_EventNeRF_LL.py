import os

# 设定帧率 (FPS)
fps = 400

# 目标文件夹路径（这里假设你需要处理多个 seq）
base_path = "H:\\Dataset\\EventNeRF\\"
# base_path = r'I:\Dataset\EventNeRF\data\nerf'  # 这是基本路径
seq_list = ["mic"]
# seq_list = ["lego"]
# seq_list = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic"]  # 7个seq文件夹

# 遍历每个序列
for seq in seq_list:
    # 假设每个 seq 下有 r_{i} 文件夹，每个文件夹包含10张图像
    for i in range(1001):
    # if 1:
    #     i=86
        # target_sequence_path = os.path.join(base_path, seq, 'train', 'Exposure', 'LL0.25', 'rgb_aperture', f'r_{i:05d}')  # 每个文件夹路径nue
        target_sequence_path = os.path.join(base_path, seq, 'train', 'Exposure', 'OverExp', 'rgb_aperture',
                                            f'r_{i:05d}')  # 每个文件夹路径nue

        # 获取文件夹中的所有图像文件（假设每个文件夹有 10 张图片）
        img_files = [f for f in sorted(os.listdir(target_sequence_path)) if f.endswith('.png')]

        # 计算时间戳
        timestamps = []
        for i in range(len(img_files)):
            timestamp = int((i / fps) * 1e6)  # 转换为微秒
            timestamps.append(timestamp)

        # 生成 info.txt 文件路径
        info_file_path = os.path.join(target_sequence_path, 'info.txt')

        # 写入 info.txt 文件
        with open(info_file_path, 'w') as info_file:
            for img_file, timestamp in zip(img_files, timestamps):
                img_file_path = os.path.join(target_sequence_path, img_file)
                info_file.write(f"{img_file_path} {timestamp:012d}\n")

        print(f"已生成 {info_file_path}")
