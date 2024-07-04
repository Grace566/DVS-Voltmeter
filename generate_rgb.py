import os
import pickle as pkl
import numpy as np
import cv2
import glob
import shutil
from tqdm import tqdm

if __name__ == '__main__':

    Dataset_dir = "I:/Dataset/3DPW"
    raw_dir = Dataset_dir + "/3DPW_origin/sequenceFiles"
    input_dir = Dataset_dir + "/3DPW_vid2e/imageFiles"
    output_dir = Dataset_dir + "/Event3DPW"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Obtain train_val_test lists
    train_list = []
    # val_list = []
    test_list = []
    for split_list in os.listdir(raw_dir):
        split_dir = raw_dir + '/' + split_list + '/'
        if split_list == "train":
            for file in os.listdir(split_dir):
                train_list.append(file.split(".")[0])
        # elif split_list == "validation":
        #     for file in os.listdir(split_dir):
        #         val_list.append(file.split(".")[0])
        elif split_list == "test":
            for file in os.listdir(split_dir):
                test_list.append(file.split(".")[0])

    # generate corresponding rgb
    print('crop data ...')
    pbar = tqdm(total=len(os.listdir(input_dir)))
    for Seq_list in os.listdir(input_dir):

        Seqinpath = input_dir + '/' + Seq_list + '/imgs/'

        if Seq_list in train_list:
            Seqoutpath = output_dir + '/train/rgb_data'
        # elif Seq_list in val_list:
        #     Seqoutpath = output_dir + '/validation/rgb_data'
        elif Seq_list in test_list:
            Seqoutpath = output_dir + '/test/rgb_data'

        if not os.path.exists(Seqoutpath):
            os.makedirs(Seqoutpath)

        for img_list in os.listdir(Seqinpath):
            img = cv2.imread(Seqinpath + img_list)
            height, width = img.shape[0], img.shape[1]
            if (height == 480) and (width == 270):
                img = img[:, 7:263, :]
            elif (height == 270) and (width == 480):
                img = img[7:263, :, :]

            cv2.imwrite((Seqoutpath + "/" + Seq_list + '_' + img_list[6:-4].zfill(6) + ".png"), img)
        pbar.update(1)

    pbar.close()

    # copy data for multi people
    print('copy data ...')
    for file in os.listdir(output_dir):      # train/test
        if os.path.splitext(file)[1] != '.zip' and os.path.splitext(file)[0] != 'validation':
            labelpath = os.path.join(output_dir, file, 'label')
            datapath = os.path.join(output_dir, file, 'rgb_data')
            if not os.path.exists(datapath):
                os.makedirs(datapath)

            print('copy ' + file + ' data ...')
            pbar = tqdm(total=len(os.listdir(datapath)))
            for data in os.listdir(datapath):
                labelname = labelpath + '\\' + data.split('.')[0] + '*.npy'
                dataname = datapath + '\\' + data
                id_num = len(glob.glob(labelname))
                if id_num > 0:      # label中存在该数据
                    base_name = datapath + '\\' + data.split('.')[0] + '_' + str(0) + '.png'
                    for id in range(id_num):
                        if id == 0:
                            os.rename(dataname, base_name)
                        else:
                            id_name = datapath + '\\' + data.split('.')[0] + '_' + str(id) + '.png'
                            shutil.copyfile(base_name, id_name)
                elif id_num == 0:   # label中没有该数据
                    os.remove(dataname)

                pbar.update(1)

            pbar.close()
