import os
import shutil
import pickle as pkl
import numpy as np
import cv2


def merge_event(event_stamp, img_stamp, i, events_path):    # Merge event according to the current timestamp
    start_stamp = img_stamp[i]
    end_stamp = img_stamp[i + 2]
    indexes = np.array(np.where((event_stamp >= start_stamp) * (event_stamp <= end_stamp))).reshape(-1)
    event = np.load((events_path + '/' + str(indexes[0]).zfill(10) + '.npz'), allow_pickle=True)
    event_x = event[event.files[0]]
    event_y = event[event.files[1]]
    event_t = event[event.files[2]]
    event_p = event[event.files[3]]
    count = 0
    for index in indexes:
        if count > 0:
            event_index = np.load((events_path + '/' + str(index).zfill(10) + '.npz'), allow_pickle=True)
            event_x = np.append(event_x, event_index[event_index.files[0]])
            event_y = np.append(event_y, event_index[event_index.files[1]])
            event_t = np.append(event_t, event_index[event_index.files[2]])
            event_p = np.append(event_p, event_index[event_index.files[3]])
        count += 1
    return event_x, event_y, event_t, event_p


if __name__ == '__main__':

    Dataset_dir = "I:/Dataset/3DPW"
    raw_dir = Dataset_dir + "/3DPW_origin/sequenceFiles"
    img_dir = Dataset_dir + "/3DPW_vid2e/imageFiles"
    events_dir = Dataset_dir + "/3DPW_vid2e/events"
    timestamp_dir = Dataset_dir + "/3DPW_vid2e/imageFiles_Upsample"
    output_dir = Dataset_dir + "/Event3DPW"

    # Obtain train_val_test lists
    train_list = []
    val_list = []
    test_list = []
    for split_list in os.listdir(raw_dir):
        split_dir = raw_dir + '/' + split_list + '/'
        if split_list == "train":
            for file in os.listdir(split_dir):
                train_list.append(file.split(".")[0])
        elif split_list == "validation":
            for file in os.listdir(split_dir):
                val_list.append(file.split(".")[0])
        elif split_list == "test":
            for file in os.listdir(split_dir):
                test_list.append(file.split(".")[0])

    # Data/Events
    fps = 30
    for event_list in os.listdir(events_dir):

        event_path = events_dir + '/' + event_list

        if event_list in train_list:
            output_path = output_dir + '/train/data'
        elif event_list in val_list:
            output_path = output_dir + '/validation/data'
        elif event_list in test_list:
            output_path = output_dir + '/test/data'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        img_len = len(os.listdir((img_dir + '/' + event_list + '/imgs')))
        img_stamp = np.arange(0, img_len * (1 / fps), 1 / fps)
        f = open((timestamp_dir + '/' + event_list + '/timestamps.txt'), "r")
        text = f.readlines()
        event_stamp = np.array([line.strip("\n") for line in text], dtype=np.float)[0:-1]
        f.close()
        for i in range(img_len - 2):        # Timestamps for each label in the sequence
            event_x, event_y, event_t, event_p = merge_event(event_stamp, img_stamp, i, event_path)
            event = np.concatenate((event_x.reshape(-1, 1), event_y.reshape(-1, 1), event_t.reshape(-1, 1),
                                    event_p.reshape(-1, 1)), axis=1)
            np.save((output_path + '/' + event_list + '_' + str(i+1).zfill(6)+'.npy'), event)
