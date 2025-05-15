import argparse
import os
import numpy as np
import cv2
import tqdm
from src.config import cfg
from src.simulator import EventSim
from src.visualize import events_to_voxel_grid, visual_voxel_grid

# event_data = np.loadtxt(r'H:\Dataset\ZJU-MoCap_pre\CoreView_377\Event\000001.txt')
# print(event_data.shape)

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='Parser of Runner of Network')
    parser.add_argument('--camera_type', type=str, help='Camera type, such as DVS346', default='DVS346')
    parser.add_argument('--model_para', type=float, nargs='+', help='Set parameters for a specific camera type', default=None)
    parser.add_argument('--seqs', type=str, nargs='+',
                        default=['CoreView_377', 'CoreView_386', 'CoreView_387',
                                 'CoreView_392', 'CoreView_393', 'CoreView_394'])
    parser.add_argument('--Dataset_dir', type=str, help='Set dataset root_path', default=None)
    args = parser.parse_args()
    return args


def integrate_cfg(cfg, command_line_args):
    args = command_line_args
    cfg.SENSOR.CAMERA_TYPE = args.camera_type if args.camera_type is not None else cfg.SENSOR.CAMERA_TYPE
    cfg.SENSOR.K = args.model_para if args.model_para is not None else cfg.SENSOR.K
    cfg.DIR.DATASET_ROOT = args.Dataset_dir if args.Dataset_dir is not None else cfg.DIR.DATASET_ROOT
    cfg.SEQUENCES = args.seqs if args.seqs else cfg.SEQUENCES
    if cfg.SENSOR.K is None or len(cfg.SENSOR.K) != 6:
        raise Exception('No model parameters given for sensor type %s' % cfg.SENSOR.CAMERA_TYPE)
    print(cfg)
    return cfg


def is_valid_dir(dirs):
    return os.path.exists(os.path.join(dirs, 'info.txt'))


def process_sequence(cfg, seq):
    """处理单个序列的图像生成事件数据

    Args:
        cfg: 配置对象，需包含Dataset_dir参数
        seq: 当前处理的序列名称
    """
    # 设置输入输出路径
    img_dir = os.path.join(cfg.DIR.DATASET_ROOT, seq, 'Src_Img', '1')
    event_dir = os.path.join(cfg.DIR.DATASET_ROOT, seq, "Event")

    # 创建输出目录（如果不存在）
    os.makedirs(event_dir, exist_ok=True)

    # 验证输入目录是否存在
    if not os.path.exists(img_dir):
        print(f"Warning: 输入目录不存在 {img_dir}")
        return

    # 读取图像信息文件
    info_file = os.path.join(img_dir, 'info.txt')
    if not os.path.exists(info_file):
        print(f"Warning: info.txt 不存在于 {img_dir}")
        return

    # 读取info.txt获取图像信息
    with open(info_file, 'r') as f:
        file_info = [line.strip().split() for line in f if line.strip()]

    # 初始化事件模拟器
    sim = EventSim(cfg=cfg, output_folder=event_dir, video_name=seq)

    # 处理每张图像
    events = []
    for img_name, timestamp_us in tqdm.tqdm(file_info, desc=f"Processing {seq}"):
        img_path = os.path.join(img_dir, img_name)

        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            print(f"Warning: 图像文件缺失 {img_path}")
            continue

        # 读取图像并生成事件
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图像 {img_path}")

        # events = sim.generate_events(image, int(timestamp_us))
        # # 保存事件数据
        # # 保存事件数据（无论是否有事件都生成文件）
        # event_filename = f"{os.path.splitext(img_name)[0]}.txt"
        # file_path = os.path.join(event_dir, event_filename)
        #
        # if events is not None and len(events) > 0:
        #     # 保存有效事件数据
        #     np.savetxt(
        #         file_path,
        #         events,  # [timestamp, x, y, polarity]
        #         fmt='%d %d %d %d',
        #         delimiter=' ',
        #         header='',
        #         comments=''
        #     )
        # else:
        #     with open(file_path, 'w') as f:
        #         f.write("")  # 写入空内容

        event = sim.generate_events(image, int(timestamp_us))

        if event is not None:
            events.append(event)

    events = np.concatenate(events, axis=0)

    # 生成所有事件后，添加时间窗口重分配逻辑
    if len(events) > 0:
        # 1. 构建图像时间点列表（单位：微秒）
        image_timestamps = [int(t) for _, t in file_info]

        # 2. 计算每个图像对应的中点时间窗口
        image_windows = []
        for i in range(len(image_timestamps)):
            if i == 0:
                prev_mid = image_timestamps[0] - (image_timestamps[1] - image_timestamps[0]) / 2
            else:
                prev_mid = (image_timestamps[i - 1] + image_timestamps[i]) / 2

            if i == len(image_timestamps) - 1:
                next_mid = image_timestamps[-1] + (image_timestamps[-1] - image_timestamps[-2]) / 2
            else:
                next_mid = (image_timestamps[i] + image_timestamps[i + 1]) / 2

            image_windows.append((prev_mid, next_mid))

        # 3. 为每个图像创建独立的事件文件
        for img_idx, (img_name, _) in enumerate(file_info):
            window_start, window_end = image_windows[img_idx]

            # 筛选属于当前窗口的事件
            mask = (events[:, 0] >= window_start) & (events[:, 0] < window_end)
            window_events = events[mask]

            if len(window_events) > 0:
                # 提取时间戳列（假设window_events是[N,4]数组，第0列为时间戳）
                timestamps = window_events[:, 0].astype(float)

                # 窗口内归一化公式：$$ t_{\text{norm}} = \frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}} $$
                normalized_t = (timestamps - window_start) / (window_end - window_start)

                # 替换时间戳列（保留其他列不变）
                window_events[:, 0] = normalized_t

            # 保存事件文件
            event_filename = f"{os.path.splitext(img_name)[0]}.txt"
            file_path = os.path.join(event_dir, event_filename)

            if len(window_events) > 0:
                np.savetxt(
                    file_path,
                    window_events,
                    fmt='%d %d %d %d',
                    delimiter=' ',
                    header='',
                    comments=''
                )
            else:
                with open(file_path, 'w') as f:
                    f.write("")


if __name__ == "__main__":
    args = get_args_from_command_line()
    cfg = integrate_cfg(cfg, args)


    for seq in args.seqs:
        print(f"\n开始处理序列: {seq}")
        process_sequence(cfg, seq)




