import os
from tqdm import tqdm

# 设定帧率 (FPS)
fps = 400  # 33  # 可以根据实际情况调整FPS

# 目标文件夹路径（这里假设你需要处理多个 seq）
# base_path = r'I:\Dataset\EventNeRF\data\nerf'  # 这是基本路径
base_path = "H:\\Dataset\\EventNeRF\\"
# seq_list = ["chair"]
# seq_list = ["drums", "ficus", "hotdog", "lego", "materials", "mic"]
seq_list = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic"]  # 7个seq文件夹

# 遍历每个序列
# for seq in seq_list:
for seq in tqdm(seq_list, desc="Processing Sequences"):

    for i in tqdm(range(1001), desc=f"Processing Images in {seq}", leave=False):
    # for i in range(1001):
        target_sequence_path = os.path.join(base_path, seq, 'train', "Exposure", "Ori", 'rgb_aperture', f'r_{i:05d}')  # 每个文件夹路径
        if not os.path.exists(target_sequence_path):
            print(f"文件夹 {target_sequence_path} 不存在!")
            continue

        # 获取文件夹中的所有图像文件
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

        # print(f"已生成 {info_file_path}")
