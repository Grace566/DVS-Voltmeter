import os
import numpy as np
from tqdm import tqdm

def convert_txt_to_npy(txt_path, npy_path):
    """
    将 txt 文件转换为 npy 格式以加速后续读取。

    参数:
        txt_path (str): 输入的 txt 文件路径。
        npy_path (str): 输出的 npy 文件路径。
    """
    try:
        # 使用 NumPy 加载 txt 数据，指定分隔符为空格
        data = np.loadtxt(txt_path, delimiter=' ', dtype=np.int32)  # 数据为整数时使用 np.int32
        # 保存为 npy 文件
        np.save(npy_path, data)
        # print(f"成功将 {txt_path} 转换为 {npy_path}")
    except Exception as e:
        print(f"转换失败: {e}")

def batch_convert_txt_to_npy(seq, base_path):
    """
    批量转换 `seq` 文件夹中的所有 .txt 文件为 .npy 格式。

    参数:
        seq (str): 序列名称，例如 'seq1', 'seq2' 等。
        base_path (str): 基础路径，指向存放数据的文件夹。
    """
    # 输入输出文件夹路径
    input_folder = os.path.join(base_path, seq, 'train', "Exposure", "Ori", 'exposure_events')
    output_folder = os.path.join(base_path, seq, 'train', "Exposure", "Ori", 'exposure_events_npy')

    # 如果输出文件夹不存在，创建该文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有 txt 文件
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    # 遍历每个 txt 文件，进行转换
    # for txt_file in txt_files:
    for txt_file in tqdm(txt_files, desc=f"Processing Images in {seq}", leave=False):
        txt_path = os.path.join(input_folder, txt_file)
        npy_file = txt_file.replace('.txt', '.npy')  # 输出的 npy 文件名
        npy_path = os.path.join(output_folder, npy_file)

        # 调用转换函数
        convert_txt_to_npy(txt_path, npy_path)

# 示例使用
# base_path = r"I:\Dataset\EventNeRF\data\nerf"  # 基础路径
base_path = "H:\\Dataset\\EventNeRF\\"
# seq_list = ["chair"]
# seq_list = ["drums", "ficus", "hotdog", "lego", "materials", "mic"]
seq_list = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic"]  # 7个seq文件夹

# 批量处理每个 seq
# for seq in seq_list:
for seq in tqdm(seq_list, desc="Processing Sequences"):
    batch_convert_txt_to_npy(seq, base_path)

