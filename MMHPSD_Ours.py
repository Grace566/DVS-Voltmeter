# # # EventHPE里面label的读取
# beta, theta, tran, joints3d, joints2d = joblib.load(
#                 '%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, end_idx))

# ############################################## 读取DHP19的点云 ################################################
# import numpy as np
# # Specify the file path
# # file_path = r'G:\Dataset\DHP19EPC\DHP19_PointCloud_testData13_17_extract_v2\data\S13_session1_mov1_frame0_cam0.npy'
# # file_path = r'G:\Dataset\DHP19EPC\DHP19_PointCloud_Dataset_extract_v2\multidim_frame_data\S1_session1_mov1_frame0_cam2.npy'
# # file_path = r'G:\Dataset\DHP19EPC\DHP19_PointCloud_Dataset_extract_v2\label\S1_session1_mov1_frame0_cam0_label.npy'
# # file_path = r'G:\Dataset\DHP19EPC\DHP19_PointCloud_Dataset_extract_v2\Point_Num_Dict.npy'
# # file_path = r'G:\Dataset\DHP19EPC\DHP19_PointCloud_Dataset_extract_v2\data\S1_session1_mov1_frame0_cam0.npy'
# # file_path = r'G:\Dataset\MMHPSD\test\multidim_frame_data\subject07_group3_time3_frame00000.npy'
# file_path = r'/mnt1/mnt/yinxiaoting/Dataset/DHP19_PointCloud_Dataset_extract_v2/multidim_frame_data/S1_session1_mov1_frame0_cam2.npy'
# point_cloud_data = np.load(file_path, allow_pickle=True)           # , allow_pickle=True
# print("Shape of point cloud data:", point_cloud_data.shape)
# print("First few points:\n", point_cloud_data[:10])

# # ################################################ Point Cloud和label可视化 ################################################
# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# def process_and_save_visualizations(events_folder, joints_folder, output_folder, sequence_name):
#     # 获取所有事件文件和关键点文件
#     event_files = [f for f in os.listdir(events_folder) if f.endswith('.npy') and sequence_name in f]
#     joint_files = [f for f in os.listdir(joints_folder) if f.endswith('.npy') and sequence_name in f]
#
#     # 确保文件按顺序处理
#     event_files.sort()
#     joint_files.sort()
#
#     # 定义关节点的连线顺序
#     skeleton = [
#         (0, 1), (1, 4), (4, 7), (7, 10),   # 右腿
#         (0, 2), (2, 5), (5, 8), (8, 11),   # 左腿
#         (0, 3), (3, 6), (6, 9),   # 躯干
#         (3, 12), (12, 15),   # 头部
#         (3, 13), (13, 16), (16, 18), (18, 20), (20, 22),   # 右臂
#         (3, 14), (14, 17), (17, 19), (19, 21), (21, 23)    # 左臂
#     ]
#
#     # 累积点云数据成事件帧
#     for event_file, joint_file in zip(event_files, joint_files):
#         # 构建完整的文件路径
#         events_path = os.path.join(events_folder, event_file)
#         joints2d_path = os.path.join(joints_folder, joint_file)
#
#         # 读取事件数据和人体关键点数据
#         events = np.load(events_path)  # (N, 4), 事件数据 (y, x, t, p)
#         joints2d = np.load(joints2d_path)  # (3, 24)
#
#         # 初始化一个空的图像帧
#         frame_height = 256
#         frame_width = 256
#         event_frame = np.zeros((frame_height, frame_width), dtype=np.uint8)
#
#         # 将事件数据填充到图像帧中
#         for event in events:
#             y, x, t, p = event
#             event_frame[int(y), int(x)] = 255 if p == 1 else 0
#
#         # 创建可视化图像
#         plt.figure(figsize=(10, 10))
#         plt.imshow(event_frame, cmap='gray')
#
#         # 绘制人体关键点和连线
#         for i in range(joints2d.shape[1]):
#             v, u, mask = joints2d[:, i]
#             plt.scatter(v, u, color='red')  # 注意：u是y坐标，v是x坐标
#
#         # 绘制连线
#         for (start, end) in skeleton:
#             u_start, v_start = joints2d[1, start], joints2d[0, start]
#             u_end, v_end = joints2d[1, end], joints2d[0, end]
#             plt.plot([v_start, v_end], [u_start, u_end], color='blue')
#
#         plt.title(f'Events and Human Keypoints Visualization - {sequence_name}')
#         plt.axis('off')
#         plt.show()
#
#         # 保存图像到指定文件夹
#         frame_number = event_file.split('_')[-1].split('.')[0]
#         output_path = os.path.join(output_folder, f'{sequence_name}_frame{frame_number}.png')
#         plt.savefig(output_path)
#         plt.close()
#
#         print(f"Visualization saved to {output_path}")
#
# # 运行示例
# events_folder = 'G:\\Dataset\\MMHPSD\\train\\data'
# joints_folder = 'G:\\Dataset\\MMHPSD\\train\\label'
# output_folder = 'G:\\Dataset\\MMHPSD\\visualizations'
# sequence_name = 'subject03_group1_time3'
#
# os.makedirs(output_folder, exist_ok=True)
# process_and_save_visualizations(events_folder, joints_folder, output_folder, sequence_name)


# # ################################################ 做数据(data) ################################################
# import os
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import cv2
#
# # Define base directories
# source_base_folder = "G:\\Dataset\\MMHPSD\\data_event_out\\data_event_raw\\data_event_raw"
# train_data_folder = "G:\\Dataset\\MMHPSD\\train\\data"
# test_data_folder = "G:\\Dataset\\MMHPSD\\test\\data"
#
# # Ensure target directories exist
# os.makedirs(train_data_folder, exist_ok=True)
# os.makedirs(test_data_folder, exist_ok=True)
#
# # Define test subjects
# test_subjects = ["subject01", "subject02", "subject07"]
#
#
# def process_csv_to_npy(src_path1, src_path2, dest_path):
#     # Read CSV file
#     data1 = pd.read_csv(src_path1)
#     data2 = pd.read_csv(src_path2)
#
#     # Convert to numpy array
#     np_data1 = data1.to_numpy()
#     np_data2 = data2.to_numpy()
#
#     # 只取前三个维度 (y, x, t)
#     np_data1_reduced = np_data1[:, :3]
#     np_data2_reduced = np_data2[:, :3]
#
#     # concat
#     np_data_reduced = np.vstack((np_data1_reduced, np_data2_reduced))
#
#     # 拓展为 (y, x, t, p)，p 维度赋值为 1
#     p = np.ones((np_data_reduced.shape[0], 1), dtype=np_data_reduced.dtype)
#     np_data_expanded = np.hstack((np_data_reduced, p))
#
#     # 增加 240，使得范围变成 0-1279
#     np_data_expanded[:, 1] += 240
#
#     # 将值范围调整为 0-255
#     def normalize_to_255(data, old_min, old_max):
#         new_min, new_max = 0, 255
#         return ((data - old_min) / (old_max - old_min) * (new_min - new_max) + new_min).astype(np.uint8)
#
#     np_data_expanded[:, 0] = normalize_to_255(np_data_expanded[:, 0], 0, 1279)
#     np_data_expanded[:, 1] = normalize_to_255(np_data_expanded[:, 1], 0, 1279)
#
#     # Save as .npy file
#     np.save(dest_path, np_data_expanded)
#
#
# # Get all subfolders
# subfolders = [f.path for f in os.scandir(source_base_folder) if f.is_dir()]
# subfolders = subfolders[145:]        # subject04_group0_time3, subject04_group2_time2没有数据， 接着断点继续生成数据
#
# for subfolder in tqdm(subfolders, desc="Processing subfolders"):
#     subfolder_name = os.path.basename(subfolder)
#     subject_id = subfolder_name.split('_')[0]
#
#     if subject_id in test_subjects:
#         data_target_base_folder = test_data_folder
#     else:
#         data_target_base_folder = train_data_folder
#
#     if 'subject04_group3_time3' in subfolder:
#         continue
#     else:
#         try:
#             # Get all CSV files in the current subfolder
#             csv_folder = os.path.join(subfolder, "event_camera", "events")
#             csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
#             csv_files.sort()
#             csv_files = csv_files[:-1]  # 最后一个文件是整个的
#             skip = 1        # 两段事件合在一起
#
#             # Process CSV files
#             for i in tqdm(range(0, len(csv_files) - skip), desc=f"Processing {subfolder_name}"):
#                 frame_num = int(csv_files[i + skip].split('event')[1].split('.csv')[0])
#                 dest_filename = f"{subfolder_name}_event{frame_num:05d}.npy"
#                 dest_path = os.path.join(data_target_base_folder, dest_filename)
#                 src_path1 = os.path.join(subfolder, 'event_camera', 'events',  csv_files[i])
#                 src_path2 = os.path.join(subfolder, 'event_camera', 'events', csv_files[i + skip])
#                 process_csv_to_npy(src_path1, src_path2, dest_path)
#         except:
#             continue
#
# print("Processing completed.")


# ################################################ 做数据(Point_Num_Dict) ################################################
import numpy as np
import os
from tqdm import tqdm


if __name__ == '__main__':
    # path of label files
    root_dir = 'G:\\Dataset\\MMHPSD\\'
    for file in ('train', 'test'):
        Point_Num_Dict = {}
        Indices_to_use = {}
        SensorSize = {}
        filename = os.path.join(root_dir, file, 'data')
        out_dir = os.path.join(root_dir, file)
        for event in tqdm(os.listdir(filename), desc=f"Processing {file} files"):
        # for event in os.listdir(filename):
            try:
                PointNum = np.load(os.path.join(filename, event),  allow_pickle=True).shape[0]
                Point_Num_Dict[event] = PointNum
            except:
                continue

        np.save(out_dir + '\\Point_Num_Dict.npy', Point_Num_Dict)

# # ################################################ 比对label和EF文件 ################################################
# import os
#
# # 定义文件夹路径
# label_folder = "G:\\Dataset\\MMHPSD\\train\\label"
# data_folder = "G:\\Dataset\\MMHPSD\\train\\multidim_frame_data"
#
# # 获取两个文件夹中的文件列表
# label_files = set(os.listdir(label_folder))
# data_files = set(os.listdir(data_folder))
#
# # 仅保留 .npy 文件
# label_files = {f for f in label_files if f.endswith('.npy')}
# data_files = {f for f in data_files if f.endswith('.npy')}
#
# # 比对差异
# label_only = label_files - data_files
# data_only = data_files - label_files
#
# # 定义一个字典来存储各个序列的统计结果
# sequences = {}
#
# for file in data_only:
#     # 提取序列信息
#     parts = file.split('_')
#     subject = parts[0]
#     group = parts[1]
#     time = parts[2]
#
#     sequence = f"{subject}_{group}_{time}"
#
#     if sequence not in sequences:
#         sequences[sequence] = 1
#     else:
#         sequences[sequence] += 1
#
# # 输出结果
# print("Files in label folder but not in data folder:")
# for file in label_only:
#     print(file)
#
# print("\nFiles in data folder but not in label folder:")
# for file in data_only:
#     print(file)



# ################################################ EF和label可视化 ################################################
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os
#
# def process_and_save_visualizations(events_folder, joints_folder, output_folder, sequence_name):
#     # 获取所有事件文件和关键点文件
#     event_files = [f for f in os.listdir(events_folder) if f.endswith('.npy')]
#     joint_files = [f for f in os.listdir(joints_folder) if f.endswith('.npy')]
#
#     # 确保文件按顺序处理
#     event_files.sort()
#     joint_files.sort()
#
#     # 定义关节点的连线顺序
#     skeleton = [
#         (0, 1), (1, 4), (4, 7), (7, 10),   # 右腿
#         (0, 2), (2, 5), (5, 8), (8, 11),   # 左腿
#         (0, 3), (3, 6), (6, 9),   # 躯干
#         (3, 12), (12, 15),   # 头部
#         (3, 13), (13, 16), (16, 18), (18, 20), (20, 22),   # 右臂
#         (3, 14), (14, 17), (17, 19), (19, 21), (21, 23)    # 左臂
#     ]
#
#     for event_file, joint_file in zip(event_files, joint_files):
#         # 构建完整的文件路径
#         events_path = os.path.join(events_folder, event_file)
#         joints2d_path = os.path.join(joints_folder, joint_file)
#
#         # 读取事件数据和人体关键点数据
#         events = np.load(events_path)  # (256, 256, 8)
#         joints2d = np.load(joints2d_path)  # (3, 24)
#
#         # 将事件数据转换为灰度图像
#         events[events > 0] = 255  # 将值为1的部分改为255
#         gray_image = np.mean(events, axis=2)
#
#         # 创建可视化图像
#         plt.figure(figsize=(10, 10))
#         plt.imshow(gray_image, cmap='gray')
#
#         # 绘制人体关键点和连线
#         for i in range(joints2d.shape[1]):
#             v, u, mask = joints2d[:, i]
#             plt.scatter(v, u, color='red')  # 注意：u是y坐标，v是x坐标
#
#         # 绘制连线
#         for (start, end) in skeleton:
#             u_start, v_start = joints2d[1, start], joints2d[0, start]
#             u_end, v_end = joints2d[1, end], joints2d[0, end]
#             plt.plot([v_start, v_end], [u_start, u_end], color='blue')
#
#         plt.title(f'Events and Human Keypoints Visualization - {sequence_name}')
#         plt.axis('off')
#
#         # 保存图像到指定文件夹
#         frame_number = event_file.split('_')[-1].split('.')[0]
#         output_path = os.path.join(output_folder, f'{sequence_name}_frame{frame_number}.png')
#         plt.savefig(output_path)
#         plt.close()
#
#         print(f"Visualization saved to {output_path}")
#
# # 定义文件夹路径
# events_folder = r'G:\Dataset\MMHPSD\test\multidim_frame_data'
# joints_folder = r'G:\Dataset\MMHPSD\test\label'
# output_folder = r'G:\Dataset\MMHPSD\GT_KeyPoints'
# sequence_name = 'subject07_group3_time3'
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# # 处理并保存所有帧的可视化图像
# process_and_save_visualizations(events_folder, joints_folder, output_folder, sequence_name)


# ################################################ 做数据(Multidim_frame_data和label) ################################################
# import os
# import cv2
# import numpy as np
# import joblib
# from tqdm import tqdm
#
# # Base directories
# source_base_folder = "G:\\Dataset\\MMHPSD\\data_event_out\\events_256"
# pose_base_folder = "G:\\Dataset\\MMHPSD\\data_event_out\\pose_events"
# train_data_folder = "G:\\Dataset\\MMHPSD\\train\\multidim_frame_data"
# test_data_folder = "G:\\Dataset\\MMHPSD\\test\\multidim_frame_data"
# train_label_folder = "G:\\Dataset\\MMHPSD\\train\\label"
# test_label_folder = "G:\\Dataset\\MMHPSD\\test\\label"
#
# # Ensure target directories exist
# os.makedirs(train_data_folder, exist_ok=True)
# os.makedirs(test_data_folder, exist_ok=True)
# os.makedirs(train_label_folder, exist_ok=True)
# os.makedirs(test_label_folder, exist_ok=True)
#
# # Define test subjects
# test_subjects = ["subject01", "subject02", "subject07"]
#
#
# def process_event_file(src_path1, src_path2, dest_path):
#     image1 = cv2.imread(src_path1, cv2.IMREAD_UNCHANGED)
#     image2 = cv2.imread(src_path2, cv2.IMREAD_UNCHANGED)
#
#     # Combine the two images along the last axis
#     combined_image = np.concatenate((image1, image2), axis=2)
#     combined_image[combined_image == 1] = 255
#
#     # combined_image = np.mean(combined_image, axis=2).astype(np.uint8)     # 放到dataloader
#
#     np.save(dest_path, combined_image)
#
#
# def process_pose_file(src_path, dest_path):
#     _, _, _, _, joints2d = joblib.load(src_path)
#     joints2d = joints2d.T
#     np.save(dest_path, joints2d)
#
#
# # Get all subfolders
# subfolders = [f.path for f in os.scandir(source_base_folder) if f.is_dir()]
# skip = 1
#
# for subfolder in tqdm(subfolders, desc="Processing subfolders"):
# # for subfolder in subfolders:
#     subfolder_name = os.path.basename(subfolder)
#     subject_id = subfolder_name.split('_')[0]
#
#     if subject_id in test_subjects:
#         data_target_base_folder = test_data_folder
#         label_target_base_folder = test_label_folder
#     else:
#         data_target_base_folder = train_data_folder
#         label_target_base_folder = train_label_folder
#
#     # Get all PNG files in the current subfolder
#     png_files = [f for f in os.listdir(subfolder) if f.endswith('.png')]
#     png_files.sort()
#
#     # Process pairs of PNG files
#     # for i in range(0, len(png_files) - skip):
#     for i in tqdm(range(0, len(png_files) - skip), desc=f"Processing {subfolder_name}"):
#         frame_num = int(png_files[i + skip].split('event')[1].split('.png')[0])
#         dest_filename = f"{subfolder_name}_frame{frame_num:05d}.npy"
#         dest_path = os.path.join(data_target_base_folder, dest_filename)
#         src_path1 = os.path.join(subfolder, png_files[i])
#         src_path2 = os.path.join(subfolder, png_files[i + skip])
#         process_event_file(src_path1, src_path2, dest_path)
#
# # Process pose files similarly
# pose_subfolders = [f.path for f in os.scandir(pose_base_folder) if f.is_dir()]
#
# for pose_subfolder in tqdm(pose_subfolders, desc="Processing pose subfolders"):
# # for pose_subfolder in pose_subfolders:
#     subfolder_name = os.path.basename(pose_subfolder)
#     subject_id = subfolder_name.split('_')[0]
#
#     if subject_id in test_subjects:
#         label_target_base_folder = test_label_folder
#     else:
#         label_target_base_folder = train_label_folder
#
#     # 对应的事件文件夹路径
#     event_folder = os.path.join(source_base_folder, subfolder_name)
#
#     # 检查事件文件夹中是否有PNG文件
#     png_files = [f for f in os.listdir(event_folder) if f.endswith('.png')]
#     if not png_files:
#         print(f"No PNG files found in {event_folder}, skipping.")
#         continue
#
#     pose_files = [f for f in os.listdir(pose_subfolder) if f.endswith('.pkl') and '_info' not in f]
#     pose_files.sort()
#
#     # for file in pose_files[1:-1]:
#     for file in tqdm(pose_files[1:-1], desc=f"Processing poses for {subfolder_name}"):
#         frame_num = int(file.split('pose')[1].split('.pkl')[0])
#         dest_filename = f"{subfolder_name}_frame{frame_num:05d}.npy"
#         dest_path = os.path.join(label_target_base_folder, dest_filename)
#         src_path = os.path.join(pose_subfolder, file)
#         process_pose_file(src_path, dest_path)




