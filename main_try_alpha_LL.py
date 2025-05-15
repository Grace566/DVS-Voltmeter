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

exposure_time = 250e3
hot_pixel_num = 5


import argparse
import os
import numpy as np
import cv2
import tqdm
from src.config import cfg
from src.simulator import EventSim
from src.visualize import events_to_voxel_grid, visual_voxel_grid
import math

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='Parser of Runner of Network')
    parser.add_argument('--camera_type', type=str, help='Camera type, such as DVS346', default='DVS346')
    parser.add_argument('--model_para', type=float, nargs='+', help='Set parameters for a specific camera type', default=None)
    parser.add_argument('--input_dir', type=str, help='Set dataset root_path', default=None)
    parser.add_argument('--output_dir', type=str, help='Set output path', default=None)
    args = parser.parse_args()
    return args

def integrate_cfg(cfg, command_line_args):
    args = command_line_args
    cfg.SENSOR.CAMERA_TYPE = args.camera_type if args.camera_type is not None else cfg.SENSOR.CAMERA_TYPE
    cfg.SENSOR.K = args.model_para if args.model_para is not None else cfg.SENSOR.K
    cfg.DIR.IN_PATH = args.input_dir if args.input_dir is not None else cfg.DIR.IN_PATH
    cfg.DIR.OUT_PATH = args.output_dir if args.output_dir is not None else cfg.DIR.OUT_PATH
    if cfg.SENSOR.K is None or len(cfg.SENSOR.K) != 6:
        raise Exception('No model parameters given for sensor type %s' % cfg.SENSOR.CAMERA_TYPE)
    print(cfg)
    return cfg

def is_valid_dir(dirs):
    return os.path.exists(os.path.join(dirs, 'info.txt'))

def calculate_psnr(img1, img2):
    """
    计算两张图片的 PSNR (峰值信噪比)

    参数:
        img1 (numpy.ndarray): 第一张图片，范围 [0, 255]。
        img2 (numpy.ndarray): 第二张图片，范围 [0, 255]。

    返回:
        float: PSNR 值。
    """
    # 计算均方误差（MSE）
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 完全相同的图像，PSNR是无穷大
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


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


def optimize_gamma(show_pic, target_img, mask, gamma_range=(1.0, 2.0), gamma_step=0.1):
    """
    优化伽玛值，通过计算 PSNR 找到最优的伽玛值。

    参数:
        events_path (str): 输入的事件数据文件路径 (.npy)。
        target_image_path (str): 目标图像路径，用于计算指标。
        gamma_range (tuple): 伽玛值的范围。
        gamma_step (float): 伽玛值的步长。

    返回:
        float: 最优的伽玛值。
    """

    best_psnr = -float('inf')
    best_gamma = gamma_range[0]

    # 遍历不同的伽玛值
    gamma = gamma_range[0]
    while gamma <= gamma_range[1]:

        # 应用伽玛校正
        fg_pic = adjust_gamma(show_pic, gamma=gamma)

        # # 将像素值为 0 的位置设置为 159/255
        # gamma_pic = np.where(fg_pic == 0, 159 / 255, fg_pic)
        gamma_pic = fg_pic
        # gamma_pic[mask == 1] = (159 / 255)

        # 计算 PSNR
        current_psnr = calculate_psnr(target_img * 255, gamma_pic * 255)  # 转换回 [0, 255]

        # 选择 PSNR 最大的伽玛值
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_gamma = gamma
            best_pic = gamma_pic
            best_fg = fg_pic

        gamma += gamma_step

    return best_pic, best_gamma, best_psnr, best_fg


def apply_log_transform(img, c=1.0):
    """
    应用对数变换调整图像对比度。

    参数:
        img (numpy.ndarray): 输入的灰度图像，范围为 [0, 1]。
        c (float): 对数变换因子。

    返回:
        numpy.ndarray: 对数变换后的图像。
    """
    img_log = c * np.log1p(img)  # log(1 + img)
    img_log = img_log / np.max(img_log)  # 归一化到 [0, 1]
    return img_log


def optimize_log_c(show_pic, target_img, mask, c_range=(1.0, 2.0), c_step=0.1):

    best_psnr = -float('inf')
    best_c = c_range[0]

    # 遍历不同的伽玛值
    c = c_range[0]
    while c <= c_range[1]:

        # 应用对数变换
        transformed_image = apply_log_transform(show_pic, c=c)

        # 将像素值为 0 的位置设置为 159/255
        # optimal_pic = np.where(transformed_image == 0, 159 / 255, transformed_image)
        optimal_pic = transformed_image
        optimal_pic[mask == 1] = (159 / 255)

        # 计算 PSNR
        current_psnr = calculate_psnr(target_img * 255, optimal_pic * 255)

        # 更新最佳 PSNR 和对应的 c
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_c = c
            best_pic = optimal_pic
            best_fg = transformed_image

        c += c_step

    return best_pic, best_c, best_psnr, best_fg

def process_dir(cfg, file_info, video_name):
    indir = os.path.join(cfg.DIR.IN_PATH, video_name)
    outdir = os.path.join(cfg.DIR.OUT_PATH, video_name)
    print(f"Processing folder {indir}... Generating events in file {outdir}")

    # file info
    file_timestamps_us = [int(info_i.split()[1]) for info_i in file_info]
    file_paths = [info_i.split()[0] for info_i in file_info]

    # set simulator
    sim = EventSim(cfg=cfg, output_folder=cfg.DIR.OUT_PATH, video_name=video_name)

    # 获取参考灰度图像（例如 r_00000.png）
    train_path = cfg.DIR.IN_PATH.split('/Exposure')[0]
    reference_image_path = os.path.join(train_path, 'grayscale_gt_mask', 'r_00473.png')
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = reference_image.astype(np.float32) / 255.0  # 转换为 [0, 1] 范围

    # 获取mask
    mask_path = os.path.join(train_path, 'rgb_mask', 'r_00473.png')
    reference_mask_img = cv2.imread(mask_path)
    mask = np.all(reference_mask_img == [0, 0, 0], axis=-1).astype(np.uint8)
    mask[mask == 1] = 1

    best_psnr = -1
    best_alpha = 1.0
    best_gray_image = None

    for alpha in np.arange(0.1, 5.0, 0.1):
    # if 1:
    #     alpha = 4
        print(f"Testing alpha = {alpha}")

        # process
        pbar = tqdm.tqdm(total=len(file_paths))
        num_events, num_on_events, num_off_events = 0, 0, 0
        events = []

        for i in range(0, len(file_paths)):
            timestamp_us = file_timestamps_us[i]
            image = cv2.imread(file_paths[i], cv2.IMREAD_GRAYSCALE)

            # event generation!!!
            event = sim.generate_events(image, timestamp_us, alpha=alpha)

            if event is not None:

                events.append(event)
                num_events += event.shape[0]
                num_on_events += np.sum(event[:, -1] == 1)
                num_off_events += np.sum(event[:, -1] == 0)

            pbar.update(1)

        events = np.concatenate(events, axis=0)

        # 重置模拟器
        sim.reset()

        # 将事件转换为灰度图像
        gray_image = events2gray(events)

        # # 不矫正
        # # optimal_pic = np.where(gray_image == 0, 159 / 255, gray_image)
        # optimal_pic = gray_image
        # optimal_pic[mask==1] = (159 / 255)

        # gamma 矫正
        optimal_pic, optimal_gamma, optimal_psnr, _ = optimize_gamma(gray_image, reference_image, mask, gamma_range=(1.0, 2.0),
                                                               gamma_step=0.1)
        # optimal_pic, optimal_gamma, optimal_psnr, _ = optimize_gamma(gray_image, reference_image, mask, gamma_range=(1.9, 2.0),
        #                                                       gamma_step=0.1)

        # # log 矫正
        # optimal_pic, optimal_c, optimal_psnr, _ = optimize_log_c(gray_image, reference_image, mask, c_range=(0.1, 6.0), c_step=0.1)

        # # 先log矫正，再gamma矫正
        # optimal_pic, optimal_c, optimal_psnr, optimal_fg = optimize_log_c(gray_image, reference_image, mask, c_range=(0.1, 6.0),
        #                                                       c_step=0.1)
        # optimal_pic, optimal_gamma, optimal_psnr, _ = optimize_gamma(optimal_fg, reference_image, mask, gamma_range=(1.0, 2.0),
        #                                                           gamma_step=0.1)

        # # 先gamma矫正，再log矫正
        # optimal_pic, optimal_gamma, optimal_psnr, optimal_fg = optimize_gamma(gray_image, reference_image, mask,
        #                                                              gamma_range=(1.0, 2.0),
        #                                                              gamma_step=0.1)
        # optimal_pic, optimal_c, optimal_psnr, _ = optimize_log_c(optimal_fg, reference_image, mask,
        #                                                                   c_range=(0.1, 6.0),
        #                                                                   c_step=0.1)

        # 计算 PSNR 值
        current_psnr = calculate_psnr(reference_image * 255, optimal_pic * 255)

        # 如果当前 alpha 对应的 PSNR 值更高，更新最佳结果
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_alpha = alpha
            best_gray_image = optimal_pic

    #######################  gamma矫正  ########################
    # 输出最佳 alpha 和对应的灰度图像
    print(f"Best alpha: {best_alpha}, Best PSNR: {best_psnr}, Best gamma: {optimal_gamma}")

    # 将最佳灰度图像保存为 PNG 文件
    # 将最佳灰度图像保存为 PNG 文件，并将最佳 alpha 和 PSNR 放入文件名
    output_gray_image_path = os.path.join(cfg.DIR.OUT_PATH,
                                          f"{video_name}_alpha_{best_alpha:.2f}_gamma_{optimal_gamma:.2f}_psnr_{best_psnr:.2f}_best_gray_image.png")
    # output_gray_image_path = os.path.join(cfg.DIR.OUT_PATH, video_name + '_best_gray_image.png')
    cv2.imwrite(output_gray_image_path, (best_gray_image * 255).clip(0, 255).astype(np.uint8))
    ########################  gamma矫正  ########################

    # ########################  log矫正  ########################
    # # 输出最佳 alpha 和对应的灰度图像
    # print(f"Best alpha: {best_alpha}, Best PSNR: {best_psnr}, Best c: {optimal_c}")
    #
    # # 将最佳灰度图像保存为 PNG 文件
    # # 将最佳灰度图像保存为 PNG 文件，并将最佳 alpha 和 PSNR 放入文件名
    # output_gray_image_path = os.path.join(cfg.DIR.OUT_PATH,
    #                                       f"{video_name}_alpha_{best_alpha:.2f}_c_{optimal_c:.2f}_psnr_{best_psnr:.2f}_best_gray_image.png")
    # # output_gray_image_path = os.path.join(cfg.DIR.OUT_PATH, video_name + '_best_gray_image.png')
    # cv2.imwrite(output_gray_image_path, (best_gray_image * 255).clip(0, 255).astype(np.uint8))
    # ########################  log矫正  ########################

    # ########################  先log矫正，再gamma矫正  ########################
    # # 输出最佳 alpha 和对应的灰度图像
    # print(f"Best alpha: {best_alpha}, Best PSNR: {best_psnr}, Best c: {optimal_c}, Best gamma: {optimal_gamma}")
    #
    # # 将最佳灰度图像保存为 PNG 文件
    # # 将最佳灰度图像保存为 PNG 文件，并将最佳 alpha 和 PSNR 放入文件名
    # output_gray_image_path = os.path.join(cfg.DIR.OUT_PATH,
    #                                       f"{video_name}_alpha_{best_alpha:.2f}_c_{optimal_c:.2f}_gamma_{optimal_gamma:.2f}_psnr_{best_psnr:.2f}_best_gray_image.png")
    # # output_gray_image_path = os.path.join(cfg.DIR.OUT_PATH, video_name + '_best_gray_image.png')
    # cv2.imwrite(output_gray_image_path, (best_gray_image * 255).clip(0, 255).astype(np.uint8))
    # ########################  log矫正  ########################

    # 重置模拟器
    sim.reset()

if __name__ == "__main__":
    args = get_args_from_command_line()
    cfg = integrate_cfg(cfg, args)

    # 获取第一个视频文件夹
    video_list = sorted(os.listdir(cfg.DIR.IN_PATH))
    if len(video_list) > 0:
        # video_i = video_list[0]  # 只处理第一个视频文件夹
        video_i = video_list[473]  # 只处理第96个视频文件夹
        video_i_path = os.path.join(cfg.DIR.IN_PATH, video_i)
        os.makedirs(os.path.join(cfg.DIR.OUT_PATH, video_i), exist_ok=True)

        if is_valid_dir(video_i_path):
            # video info
            with open(os.path.join(cfg.DIR.IN_PATH, video_i, 'info.txt'), 'r') as f:
                video_info = f.readlines()
            # simulation
            process_dir(cfg=cfg, file_info=video_info, video_name=video_i)
    else:
        print("No video folders found in the input directory.")
