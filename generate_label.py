import os
import pickle as pkl
import numpy as np
import cv2


if __name__ == '__main__':

    Dataset_dir = "E:/DVS-SIM/src"
    label_dir = Dataset_dir + "/3DPW_origin/sequenceFiles"
    output_dir = Dataset_dir + "/Event3DPW"
    upsampled_dir = Dataset_dir + "/3DPW_vid2e/imageFiles_Upsample"

    # Resize pose2D it according to the size of upsampled image
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            path = os.path.join(root, file)
            seq = pkl.load(open(path, 'rb'), encoding='bytes')

            num_people = len(seq[b'poses'])
            num_frames = len(seq[b'img_frame_ids'])
            assert (seq[b'poses2d'][0].shape[0] == num_frames)

            for p_id in range(num_people):
                label = seq[b'poses2d'][p_id]
                img_path = upsampled_dir + '/' + file.split(".")[0] + '/imgs/00000000.png'
                img = cv2.imread(img_path)
                height, width = img.shape[0], img.shape[1]
                cropsize = 7
                if (height == 480) and (width == 256):
                    label[:, 0, :] = np.maximum(label[:, 0, :] / 4 - cropsize, 0)
                    label[:, 1, :] = label[:, 1, :] / 4
                elif (height == 256) and (width == 480):
                    label[:, 0, :] = label[:, 0, :] / 4
                    label[:, 1, :] = np.maximum(label[:, 1, :] / 4 - cropsize, 0)

                for i in range(label.shape[0] - 2):
                    single_label = label[i + 1, :, :]
                    output_path = output_dir + '/' + root.split('\\')[-1] + '/label/'
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    np.save((output_path + file.split(".")[0] + '_' + str(i + 1).zfill(6)
                             + '_' + str(p_id) + '.npy'), single_label)
