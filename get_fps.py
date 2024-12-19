import cv2

# 替换为你的 MP4 文件路径
video_path = 'F:\Dataset\ZJU-MoCap\CoreView_386\camera_1.mp4'

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Cannot open video file.")
else:
    # 获取 FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

# 释放视频对象
cap.release()
