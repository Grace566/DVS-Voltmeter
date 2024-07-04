import os
import numpy as np

if __name__ == '__main__':
    Dataset_dir = "E:/DVS-SIM/src/"
    raw_dir = 'G:/Dataset/3DPW/3DPW_origin/sequenceFiles'
    img_dir = Dataset_dir + "/imageFiles"
    events_dir = Dataset_dir + "/events"
    output_dir = "E:/DVS-SIM/Event3DPW_Noise"

    # Obtain train_val_test lists
    train_list = []
    val_list = []
    test_list = []
    for split_list in os.listdir(raw_dir):
        split_dir = os.path.join(raw_dir, split_list)
        if split_list == "train":
            train_list.extend([file.split(".")[0] for file in os.listdir(split_dir)])
        elif split_list == "validation":
            val_list.extend([file.split(".")[0] for file in os.listdir(split_dir)])
        elif split_list == "test":
            test_list.extend([file.split(".")[0] for file in os.listdir(split_dir)])

    # Data/Events
    fps = 30
    for event_list in os.listdir(events_dir):
        if event_list.endswith('.txt'):

            event_list = event_list.split('.')[0]

            event_path = os.path.join(events_dir, event_list) + '.txt'

            if event_list in train_list:
                output_path = os.path.join(output_dir, 'train', 'data')
            elif event_list in val_list:
                output_path = os.path.join(output_dir, 'validation', 'data')
            elif event_list in test_list:
                output_path = os.path.join(output_dir, 'test', 'data')

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            img_len = len(os.listdir(os.path.join(img_dir, event_list, 'imgs')))
            img_stamp = np.arange(0, img_len * (1 / fps), 1 / fps)

            # Load event data
            data = np.loadtxt(event_path)

            # Iterate over img_stamp to create event npy files
            for i in range(len(img_stamp) - 2):
                start_time = img_stamp[i]
                end_time = img_stamp[i + 2]

                # Select events within the time range [start_time, end_time)
                mask = (data[:, 1] >= start_time) & (data[:, 1] < end_time)
                selected_events = data[mask]

                # Save selected_events as npy file
                np.save(os.path.join(output_path, f'{event_list}_{i + 1:06}.npy'), selected_events)
