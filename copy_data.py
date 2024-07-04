import os
import shutil
import pickle as pkl
import numpy as np
import cv2
import glob


if __name__ == '__main__':

    Dataset_dir = "I:\\Dataset\\3DPW\\Event3DPW"

    for file in os.listdir(Dataset_dir):
        labelpath = os.path.join(Dataset_dir, file, 'label')
        datapath = os.path.join(Dataset_dir, file, 'data')
        for data in os.listdir(datapath):
            labelname = labelpath + '\\' + data.split('.')[0] + '*.npy'
            dataname = datapath + '\\' + data
            id_num = len(glob.glob(labelname))
            base_name = datapath + '\\' + data.split('.')[0] + '_' + str(0) + '.npy'
            for id in range(id_num):
                if id == 0:
                    os.rename(dataname, base_name)
                else:
                    id_name = datapath + '\\' + data.split('.')[0] + '_' + str(id) + '.npy'
                    shutil.copyfile(base_name, id_name)