# -*- coding: utf-8 -*-
# @Time    : 2022/10/10 9:48
# @Author  : Jiaan Chen
import glob
import os
import numpy as np
from tqdm import tqdm


def multidim_evframe_gen(data, imageH=260, imageW=346):
    x = data[0, :]  # x
    y = data[1, :]  # y
    t = data[2, :]  # t
    p = data[3, :]  # p [0, 1]
    num_events = len(x)
    if num_events > 0:
        t_ref = t[-1]  # time of the last event in the packet
        # tau = 50000  # decay parameter (in micro seconds)
        tau = (t[-1] - t[0]) / 2
        img_size = (imageH, imageW)
        img_pos = np.zeros(img_size, np.int)
        img_neg = np.zeros(img_size, np.int)
        sae_pos = np.zeros(img_size, np.float32)
        sae_neg = np.zeros(img_size, np.float32)
        cnt = np.zeros(img_size, np.float32)
        sae = np.zeros(img_size, np.float32)
        for idx in range(num_events):
            coordx = int(x[idx])
            coordy = int(y[idx])
            if p[idx] > 0:
                img_pos[coordy, coordx] += 1  # count events
                sae_pos[coordy, coordx] = np.exp(-(t_ref - t[idx]) / tau)
            else:
                img_neg[coordy, coordx] += 1
                sae_neg[coordy, coordx] = np.exp(-(t_ref - t[idx]) / tau)
            cnt[coordy, coordx] += 1
            sae[coordy, coordx] = np.exp(-(t_ref - t[idx]) / tau)

        cnt_sae = np.multiply(cnt, sae)

        img_pos = normalizeImage3Sigma(img_pos, imageH=imageH, imageW=imageW)
        img_neg = normalizeImage3Sigma(img_neg, imageH=imageH, imageW=imageW)
        sae_pos = normalizeImage3Sigma(sae_pos, imageH=imageH, imageW=imageW)
        sae_neg = normalizeImage3Sigma(sae_neg, imageH=imageH, imageW=imageW)
        cnt_sae = normalizeImage3Sigma(cnt_sae, imageH=imageH, imageW=imageW)
        cnt = normalizeImage3Sigma(cnt, imageH=imageH, imageW=imageW)
        sae = normalizeImage3Sigma(sae, imageH=imageH, imageW=imageW)

        md_evframe = np.concatenate((img_pos[:, :, np.newaxis], img_neg[:, :, np.newaxis],
                                     sae_pos[:, :, np.newaxis], sae_neg[:, :, np.newaxis],
                                     cnt_sae[:, :, np.newaxis], cnt[:, :, np.newaxis], sae[:, :, np.newaxis]), axis=2)
        md_evframe = md_evframe.astype(np.uint8)
    else:
        img_size = (imageH, imageW)
        img_pos = np.zeros(img_size, np.int)
        img_neg = np.zeros(img_size, np.int)
        sae_pos = np.zeros(img_size, np.float32)
        sae_neg = np.zeros(img_size, np.float32)
        cnt = np.zeros(img_size, np.float32)
        sae = np.zeros(img_size, np.float32)
        cnt_sae = np.multiply(cnt, sae)

        md_evframe = np.concatenate((img_pos[:, :, np.newaxis], img_neg[:, :, np.newaxis],
                                     sae_pos[:, :, np.newaxis], sae_neg[:, :, np.newaxis],
                                     cnt_sae[:, :, np.newaxis], cnt[:, :, np.newaxis], sae[:, :, np.newaxis]), axis=2)
        md_evframe = md_evframe.astype(np.uint8)

    return md_evframe


def normalizeImage3Sigma(image, imageH=260, imageW=346):
    """followed by matlab dhp19 generate"""
    sum_img = np.sum(image)
    count_image = np.sum(image > 0)
    mean_image = sum_img / count_image
    var_img = np.var(image[image > 0])
    sig_img = np.sqrt(var_img)

    if sig_img < 0.1 / 255:
        sig_img = 0.1 / 255

    numSDevs = 3.0
    # Rectify polarity=true
    meanGrey = 0
    range_old = numSDevs * sig_img
    half_range = 0
    range_new = 255
    # Rectify polarity=false
    # meanGrey=127 / 255
    # range= 2*numSDevs * sig_img
    # halfrange = numSDevs * sig_img

    normalizedMat = np.zeros([imageH, imageW])
    for i in range(imageH):
        for j in range(imageW):
            l = image[i, j]
            if l == 0:
                normalizedMat[i, j] = meanGrey
            else:
                f = (l + half_range) * range_new / range_old
                if f > range_new:
                    f = range_new

                if f < 0:
                    f = 0
                normalizedMat[i, j] = np.floor(f)

    return normalizedMat


# # 指定文件路径
# file_path = 'G:\\Dataset\\DHP19EPC\\DHP19_PointCloud_Dataset_extract_v2\\multidim_frame_data\\S10_session1_mov1_frame0_cam2.npy'
# # 读取 .npy 文件
# data = np.load(file_path)

# # path of train data
# root_train_data_dir = '/mnt1/yinxiaoting/Event3DPW/train/'
# # path of valid data
# root_valid_data_dir = '/mnt1/yinxiaoting/Event3DPW/test/'
# path of train data
root_train_data_dir = 'G:/Dataset//DVS-SIM/Event3DPW_Noise/train/'
# path of valid data
root_valid_data_dir = 'G:/Dataset//DVS-SIM/Event3DPW_Noise/test/'

root_data_dir = root_train_data_dir + 'data//'
out_frame_dir = root_train_data_dir + 'multidim_frame_data//'
if not os.path.exists(out_frame_dir):
    os.makedirs(out_frame_dir)
root_size_dir=root_train_data_dir + 'EventSize.npy'

dvs_frames = sorted(glob.glob(os.path.join(root_data_dir, "*.npy")))
dvs_frames = dvs_frames[21756:]     # 从这断的
EventSize = np.load(root_size_dir, allow_pickle=True).item()

print('train data ...')
pbar = tqdm(total=len(dvs_frames))
for file_dir in dvs_frames:
    data_name = os.path.basename(file_dir).split('.')[0]
    try:
        data = np.load(file_dir)
        sensor_sizeW = EventSize[(data_name + '.npy')][1]
        sensor_sizeH = EventSize[(data_name + '.npy')][0]
        data[:, 0] -= 1
        data = multidim_evframe_gen(data.T, imageH=sensor_sizeH, imageW=sensor_sizeW)

        np.save(out_frame_dir + data_name + '.npy', data)

    except:     #只有'outdoors_climbing_02_000394_0' 没有eventsize
        sensor_sizeW = 256
        sensor_sizeH = 480
        data = np.zeros((sensor_sizeH, sensor_sizeW, 7)).astype(np.uint8)
        np.save(out_frame_dir + data_name + '.npy', data)

    pbar.update(1)

pbar.close()


root_data_dir = root_valid_data_dir + 'data//'
out_frame_dir = root_valid_data_dir + 'multidim_frame_data//'
if not os.path.exists(out_frame_dir):
    os.makedirs(out_frame_dir)
root_size_dir=root_valid_data_dir + 'EventSize.npy'

dvs_frames = sorted(glob.glob(os.path.join(root_data_dir, "**.npy")))
EventSize = np.load(root_size_dir, allow_pickle=True).item()

print('valid data ...')
pbar = tqdm(total=len(dvs_frames))
for file_dir in dvs_frames:
   data_name = os.path.basename(file_dir).split('.')[0]

   try:
       data = np.load(file_dir)
       sensor_sizeW = EventSize[(data_name + '.npy')][1]
       sensor_sizeH = EventSize[(data_name + '.npy')][0]
       data[:, 0] -= 1
       data = multidim_evframe_gen(data.T, imageH=sensor_sizeH, imageW=sensor_sizeW)

       np.save(out_frame_dir + data_name + '.npy', data)

   except:
       sensor_sizeW = EventSize[(data_name + '.npy')][1]
       sensor_sizeH = EventSize[(data_name + '.npy')][0]
       data = np.zeros((sensor_sizeH, sensor_sizeW, 7)).astype(np.uint8)
       np.save(out_frame_dir + data_name + '.npy', data)

   pbar.update(1)

pbar.close()



