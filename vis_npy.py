import numpy as np
import cv2

def visualize_events_as_video(event_file, width=240, height=180, time_window=None, output_video="events_video.mp4", fps=30, video_duration=10):
    """
    可视化事件数据，并将其保存为视频，视频总时长为 10 秒。正极性为红色，负极性为蓝色。

    参数：
        event_file (str): 事件数据的文件路径。
        width (int): 图像的宽度（根据传感器的分辨率）。
        height (int): 图像的高度（根据传感器的分辨率）。
        time_window (tuple): 可选，事件时间的窗口，用于筛选特定时间范围的事件。格式：(start_time, end_time)。
        output_video (str): 输出视频的文件路径。
        fps (int): 每秒显示的帧数，用于控制视频播放的速度。
        video_duration (int): 视频的总时长（秒）。默认为 10 秒。
    """
    # 1. 加载事件数据
    events = np.load(event_file)  # 加载 (N, 4) 数据，包含 [t, x, y, p]
    t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

    # 2. 如果有时间窗口，筛选时间范围内的事件
    if time_window is not None:
        start_time, end_time = time_window
        mask = (t >= start_time) & (t <= end_time)
        t, x, y, p = t[mask], x[mask], y[mask], p[mask]

    # 3. 按时间戳排序事件
    sorted_indices = np.argsort(t)
    t, x, y, p = t[sorted_indices], x[sorted_indices], y[sorted_indices], p[sorted_indices]

    # 4. 计算总帧数：视频总时长 * fps
    total_frames = fps * video_duration

    # 5. 创建 VideoWriter 对象，用于保存视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 6. 计算每帧的时间范围
    min_time = t[0]
    max_time = t[-1]
    time_per_frame = (max_time - min_time) / total_frames

    # 7. 初始化事件图像（RGB 图像）
    image = np.zeros((height, width, 3), dtype=np.uint8)  # 使用 3 通道来表示 RGB 图像

    # 8. 按帧逐步更新图像
    current_time = min_time
    frame_start_idx = 0

    for frame in range(total_frames):
        # 更新当前时间
        frame_end_time = current_time + time_per_frame

        # 在当前时间窗口内更新图像
        for i in range(frame_start_idx, len(t)):
            if t[i] > frame_end_time:
                break  # 如果事件时间大于当前时间窗口，停止更新
            # 根据极性更新图像
            if p[i] > 0:  # 正极性，使用红色
                image[int(y[i]), int(x[i])] = [0, 0, 255]  # 红色 [B, G, R]
            else:  # 负极性，使用蓝色
                image[int(y[i]), int(x[i])] = [255, 0, 0]  # 蓝色 [B, G, R]

        # 保存当前帧
        out.write(image)

        # 更新当前时间
        current_time = frame_end_time

    # 9. 释放资源
    out.release()
    print(f"视频已保存为 {output_video}")

# 示例：将事件数据可视化并保存为视频，总时长为 10 秒
event_file = r"H:\Dataset\EventNeRF\chair\train\Exposure\Ori\exposure_events_npy\r_00000.npy"
output_video = r"H:\Dataset\EventNeRF\chair\train\Exposure\Ori\events_video_red_blue_10sec.mp4"
visualize_events_as_video(event_file, width=346, height=260, fps=30, video_duration=10, output_video=output_video)
