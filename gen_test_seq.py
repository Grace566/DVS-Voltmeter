import os

if __name__ == '__main__':

    Dataset_dir = "I:/Dataset/3DPW"
    raw_dir = Dataset_dir + "/3DPW_origin/sequenceFiles"
    # output_dir = Dataset_dir + "/Event3DPW/test/"
    output_dir = Dataset_dir + "/Event3DPW/train/"

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

    with open((output_dir + 'seqname.txt'), 'w') as f:
        # for i in test_list:
        for i in train_list:
            f.write(i + '\n')

