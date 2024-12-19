import numpy as np

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
        print(f"成功将 {txt_path} 转换为 {npy_path}")
    except Exception as e:
        print(f"转换失败: {e}")

# 示例使用
txt_path = r"F:\Dataset\ZJU-MoCap_pre\CoreView_386\events\1.txt"
npy_path = r"F:\Dataset\ZJU-MoCap_pre\CoreView_386\events\1.npy"

convert_txt_to_npy(txt_path, npy_path)
