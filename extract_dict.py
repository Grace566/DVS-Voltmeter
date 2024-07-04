# -*- coding: utf-8 -*-
# @Time    : 2023/3/2 10:52
# @Author  : Xiaoting Yin, Jiaan Chen

# Generate DVS PointCloud Dataset dict recording event numbers from .pkl and save as .npy

import numpy as np
import os
import glob
import cv2

VIS_THRESH = 0.3        # Threshold for visibility.（VIBE）https://github.com/mkocabas/VIBE
MIN_KP = 6
if __name__ == '__main__':
    # path of label files
    root_dir = "E:/DVS-SIM/Event3DPW_Noise"
    img_dir = 'G:\\Dataset\\3DPW\\3DPW_vid2e\\imageFiles_Upsample'
    for file in os.listdir(root_dir):
        # if file == 'train':
        if 1:

            Point_Num_Dict = {}
            Indices_to_use = {}
            SensorSize = {}
            filename = os.path.join(root_dir, file, 'data')
            out_dir = os.path.join(root_dir, file)
            for event in os.listdir(filename):
                PointNum = np.load(os.path.join(filename, event)).shape[0]
                Point_Num_Dict[event] = PointNum

                labelname = os.path.join(root_dir, file, 'label', event)
                label = np.load(labelname)
                label = np.array([1, 2, 3])
                u, v, mask = label  # 不需要显式转换为 np.float, for numpy >= 1.20
                Indices_to_use[event] = ((mask > VIS_THRESH).sum(-1) > MIN_KP).astype(int)

                img_path = img_dir + '\\' + '_'.join(event.split('_')[0:3]) + '\\imgs\\' + \
                           event.split('.')[0].split('_')[3].zfill(8) + '.png'
                img = cv2.imread(img_path)
                SensorSize[event] = img.shape[0:2]

            np.save(out_dir + '\\Point_Num_Dict.npy', Point_Num_Dict)
            np.save(out_dir + '\\Indices_to_use.npy', Indices_to_use)
            np.save(out_dir + '\\EventSize.npy', SensorSize)
