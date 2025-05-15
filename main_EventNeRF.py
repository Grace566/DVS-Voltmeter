#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/7/12 17:30
@Author  :   Songnan Lin, Ye Ma
@Contact :   songnan.lin@ntu.edu.sg, my17@tsinghua.org.cn
@Note    :   
@inproceedings{lin2022dvsvoltmeter,
  title={DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors},
  author={Lin, Songnan and Ma, Ye and Guo, Zhenhua and Wen, Bihan},
  booktitle={ECCV},
  year={2022}
}
'''

import argparse
import os
import numpy as np
import cv2
import tqdm
from src.config import cfg
from src.simulator import EventSim
from src.visualize import events_to_voxel_grid, visual_voxel_grid

exposure_time = 250e3
hot_pixel_num = 5

def get_raw_pic(pic_events):
    pic = np.zeros((260, 346), np.int32)
    for evs in pic_events:
        if evs[3] == 1 and pic[evs[2], evs[1]] == 0:
            pic[evs[2], evs[1]] = evs[0]
    return pic

def time2gray(pic, open_time=0):            # 原版代码
    bias = 0
    pic = pic.astype(np.float32)
    pic_avg = pic[np.nonzero(pic)].mean()
    pic_std = pic[np.nonzero(pic)].std()
    pic[np.where(pic == 0)] = exposure_time+open_time
    # hot pixel removal
    # pic[199][1010] = pic_avg
    # pic[364][775] = pic_avg
    # pic[55][536] = pic_avg
    # pic[56][1052] = pic_avg
    # pic[64][413]=pic_avg
    #
    # pic[np.where(pic == np.nanmin(pic))] = pic_avg
    # pic[np.where(pic == np.nanmin(pic))] = pic_avg
    # pic[np.where(pic == np.nanmin(pic))] = pic_avg
    # pic[np.where(pic == np.nanmin(pic))] = pic_avg
    # pic[np.where(pic == np.nanmin(pic))] = pic_avg
    # pic[np.where(pic == np.nanmin(pic))] = pic_avg
    # pic[np.where(pic == np.nanmin(pic))] = pic_avg
    # pic[np.where(pic == np.nanmin(pic))] = pic_avg


    # pic[pic < pic_avg - 2 * pic_std] = pic_avg - 2 * pic_std
    # pic[pic > pic_avg + 2 * pic_std] = pic_avg + 2 * pic_std


    # pic = error_npy_manual_check(pic)

    min_index = np.where(pic == np.nanmin(pic))
    # print(min_index[0][0], min_index[1][0])
    # choose different bias for bulk brightness shift, probably not needed
    pic = pic - open_time + bias
    # plt.figure()
    # plt.hist(pic.flatten(), bins=100)
    # plt.show()

    pic = pic / exposure_time
    pic[pic > 1] = 1
    pic = 1 / np.sin(pic * np.pi / 2)**2
    # print(np.max(pic))
    # if np.max(pic) > 50:
    #     pic[min]
    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
    # pic = (pic) / (np.max(pic))
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.hist(pic.flatten(), bins=100)
    # plt.subplot(2, 1, 2)
    # plt.imshow(pic, cmap='gray')
    # plt.show()
    # cv2.namedWindow("pic", cv2.WINDOW_FREERATIO)
    # cv2.imshow("pic", pic)
    # cv2.waitKey(1)

    return pic, np.max(pic)


def events2gray(events):

    if len(events) > hot_pixel_num:
        # 为了消除一些异常早出现的事件(hot pixel)带来的影响，这里进行一个clip操作
        events[:hot_pixel_num, 0] = events[hot_pixel_num, 0]

    # 获取原始图像
    pic = get_raw_pic(events)

    # 转换为灰度图像
    show_pic, max_value = time2gray(pic)

    return show_pic

def adjust_gamma(image, gamma=1.0):
    """
    对输入图像进行伽玛校正，支持 np.float32 类型的输入并返回同样类型的输出。

    参数:
        image (numpy.ndarray): 输入图像，数据类型应为 np.float32，值域 [0, 1].
        gamma (float): 伽玛值，默认为 1.0，即不改变亮度。

    返回:
        numpy.ndarray: 经过伽玛校正后的图像，数据类型为 np.float32，值域 [0, 1].
    """
    # 确保输入图像是 float32 并且值域在 [0, 1]
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise ValueError("Input image must be a numpy array of type float32 with values in the range [0, 1].")

    # 创建伽玛校正查找表
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # 将输入图像从 float32 转换为 uint8，假设输入已经归一化到 [0, 1]
    image_uint8 = (image * 255).astype(np.uint8)

    # 应用伽玛校正
    corrected_image_uint8 = cv2.LUT(image_uint8, table)

    # 将结果转换回 float32 并归一化到 [0, 1]
    corrected_image_float32 = corrected_image_uint8.astype(np.float32) / 255.0

    return corrected_image_float32


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='Parser of Runner of Network')
    parser.add_argument('--camera_type', type=str, help='Camera type, such as DVS346', default='DVS346')
    parser.add_argument('--model_para', type=float, nargs='+', help='Set parameters for a specific camera type', default=None)
    parser.add_argument('--input_dir', type=str, help='Set dataset root_path', default=None)
    parser.add_argument('--output_dir', type=str, help='Set output path', default=None)
    parser.add_argument('--alpha', type=float, help='Alpha parameter for event generation', default=1.0)
    parser.add_argument('--gamma', type=float, help='Gamma parameter for event generation', default=1.0)

    args = parser.parse_args()
    return args

import argparse


def integrate_cfg(cfg, command_line_args):
    args = command_line_args
    cfg.SENSOR.CAMERA_TYPE = args.camera_type if args.camera_type is not None else cfg.SENSOR.CAMERA_TYPE
    cfg.SENSOR.K = args.model_para if args.model_para is not None else cfg.SENSOR.K
    cfg.DIR.IN_PATH = args.input_dir if args.input_dir is not None else cfg.DIR.IN_PATH
    cfg.DIR.OUT_PATH = args.output_dir if args.output_dir is not None else cfg.DIR.OUT_PATH

    if cfg.SENSOR.K is None or len(cfg.SENSOR.K) != 6:
        raise Exception('No model parameters given for sensor type %s' % cfg.SENSOR.CAMERA_TYPE)

    # 更新 alpha 参数
    cfg.GENERATION = cfg.get("GENERATION", {})  # 确保 cfg.GENERATION 存在
    cfg.GENERATION['ALPHA'] = args.alpha if args.alpha is not None else cfg.GENERATION.get('ALPHA', 1.0)

    # 更新 gamma 参数
    cfg.GENERATION['GAMMA'] = args.gamma if args.gamma is not None else cfg.GENERATION.get('GAMMA', 1.0)

    print(cfg)
    return cfg

def is_valid_dir(dirs):
    return os.path.exists(os.path.join(dirs, 'info.txt'))

def process_dir(cfg, file_info, video_name):
    indir = os.path.join(cfg.DIR.IN_PATH, video_name)
    outdir = os.path.join(cfg.DIR.OUT_PATH, video_name)
    print(f"Processing folder {indir}... Generating events in file {outdir}")

    # file info
    file_timestamps_us = [int(info_i.split()[1]) for info_i in file_info]
    file_paths = [info_i.split()[0] for info_i in file_info]

    # mask file
    train_path = cfg.DIR.IN_PATH.split('/Exposure')[0]
    mask_dir = os.path.join(train_path, 'rgb_mask')
    mask_path = os.path.join(mask_dir, (file_paths[0].split("\\")[8] + '.png'))
    reference_mask_img = cv2.imread(mask_path)
    mask = np.all(reference_mask_img == [0, 0, 0], axis=-1).astype(np.uint8)
    mask[mask == 1] = 1

    # mask_dir = os.path.join(train_path, 'sam_mask')
    # mask_path = os.path.join(mask_dir, (file_paths[0].split("\\")[8] + '.png'))
    # mask = cv2.imread(mask_path)
    # mask = np.all(mask == [0, 0, 0], axis=-1).astype(np.uint8)

    # set simulator
    sim = EventSim(cfg=cfg, output_folder=cfg.DIR.OUT_PATH, video_name=video_name)

    # process
    pbar = tqdm.tqdm(total=len(file_paths))
    num_events, num_on_events, num_off_events = 0, 0, 0
    events = []
    for i in range(0, len(file_paths)):
        timestamp_us = file_timestamps_us[i]
        image = cv2.imread(file_paths[i], cv2.IMREAD_GRAYSCALE)

        # event generation!!!
        event = sim.generate_events(image, timestamp_us, alpha=cfg.GENERATION['ALPHA'])

        if event is not None:

            events.append(event)
            num_events += event.shape[0]
            num_on_events += np.sum(event[:, -1] == 1)
            num_off_events += np.sum(event[:, -1] == 0)

        pbar.set_description(f"Events generated: {num_events}")
        pbar.update(1)

    events = np.concatenate(events, axis=0)
    # 将事件转换为灰度图像
    gray_image = events2gray(events)

    # 应用伽玛校正
    gray_image = adjust_gamma(gray_image, gamma=cfg.GENERATION['GAMMA'])

    # # 将像素值为 0 的位置设置为 159/255
    # gray_image = np.where(gray_image == 0, 159 / 255, gray_image)
    # gray_image[mask==1] = (159 / 255)               # TODO: 做定性图的时候忽略这步

    output_gray_image_path = os.path.join(cfg.DIR.OUT_PATH,
                                          video_name + '.png')

    cv2.imwrite(output_gray_image_path, (gray_image * 255).clip(0, 255).astype(np.uint8))

    sim.reset()


if __name__ == "__main__":
    args = get_args_from_command_line()
    cfg = integrate_cfg(cfg, args)

    video_list = sorted(os.listdir(cfg.DIR.IN_PATH))
    for video_i in video_list:
    # if 1:
    #     video_i = 'r_00096'
        video_i_path = os.path.join(cfg.DIR.IN_PATH, video_i)
        # os.makedirs(os.path.join(cfg.DIR.OUT_PATH, video_i), exist_ok=True)

        if is_valid_dir(video_i_path):
            # video info
            with open(os.path.join(cfg.DIR.IN_PATH, video_i, 'info.txt'), 'r') as f:
                video_info = f.readlines()
            # simulation
            process_dir(cfg=cfg, file_info=video_info, video_name=video_i)
