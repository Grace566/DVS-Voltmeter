import os
import subprocess

# 序列列表
# seq_list = ["lego"]
seq_list = ["mic"]
# seq_list = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic"]
# seq_list = ["drums", "ficus", "hotdog", "lego", "materials", "mic"]


# #######################  main_EventNeRF.py + lowlight ########################
# base_input_dir = "H:/Dataset/EventNeRF/{}/train/Exposure/LL0.25/rgb_aperture"
# base_output_dir = "H:/Dataset/EventNeRF/{}/train/Exposure/LL0.25/exposure_events_gray"

########################  main_EventNeRF.py + overexposure ########################
base_input_dir = "H:/Dataset/EventNeRF/{}/train/Exposure/OverExp/rgb_aperture"
base_output_dir = "H:/Dataset/EventNeRF/{}/train/Exposure/OverExp/exposure_events_gray"


# 为每个序列指定不同的 alpha 值
alpha_values = {
    "chair": 2.5,
    "drums": 2.3,
    "ficus": 1.5,
    "hotdog": 3.3,
    "lego": 2.3,
    "materials": 2.1,
    "mic": 2.1,
}

gamma_values = {
    "chair": 1.9,
    "drums": 1.9,
    "ficus": 1.9,
    "hotdog": 1.9,
    "lego": 1.9,
    "materials": 1.9,
    "mic": 1.9,
}



# 循环处理每个序列
for seq in seq_list:
    # 替换路径
    input_dir = base_input_dir.format(seq)
    output_dir = base_output_dir.format(seq)

    # 判断文件夹是否存在
    if not os.path.exists(output_dir):
        # 如果不存在，则创建文件夹
        os.makedirs(output_dir)
        print(f"Directory created: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")

    # # 构造命令行
    # command = [
    #     "python", "main_try_alpha_LL.py",
    #     # "--camera_type", "DVS346",
    #     "--input_dir", input_dir,
    #     "--output_dir", output_dir
    # ]

    # 提取当前序列的 alpha 和 gamma 值
    alpha = alpha_values.get(seq, 1.0)  # 默认值为 1.0，如果没找到对应的值
    gamma = gamma_values.get(seq, 1.0)  # 默认值为 1.0，如果没找到对应的值
    command = [
        "python", "main_EventNeRF.py",
        # "--camera_type", "DVS346",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--alpha", str(alpha),  # 传递 alpha 参数
        "--gamma", str(gamma),
    ]

    # # 构造命令行
    # command = [
    #     "python", "main.py",
    #     # "--camera_type", "DVS346",
    #     "--input_dir", input_dir,
    #     "--output_dir", output_dir
    # ]

    # 打印命令（可选）
    print(f"Executing command: {' '.join(command)}")

    # 调用 subprocess 运行命令
    subprocess.run(command, check=True)
