import os
import numpy as np
import shutil
import random

SRC_PATH = '../dataset'

LABEL_PATH = '../dataset/labels'

ROOT_PATH = '../dataset_train'

TRAIN_PATH = '../dataset_train/train'

TRAIN_LABEL_PATH = '../dataset_train/train_labels'

VAL_PATH = '../dataset_train/val'

VAL_LABEL_PATH = '../dataset_train/val_labels'

RATiO_VAL = 0.2


def main():
    file_list = []
    # shutil.rmtree(TRAIN_PATH)
    # shutil.rmtree(TRAIN_LABEL_PATH)
    # shutil.rmtree(VAL_PATH)
    # shutil.rmtree(VAL_LABEL_PATH)
    if os.path.exists(ROOT_PATH):
        shutil.rmtree(ROOT_PATH)

    os.mkdir(ROOT_PATH)
    os.mkdir(TRAIN_PATH)
    os.mkdir(TRAIN_LABEL_PATH)
    os.mkdir(VAL_PATH)
    os.mkdir(VAL_LABEL_PATH)

    for file in os.listdir(SRC_PATH):
        if os.path.isdir(os.path.join(SRC_PATH, file)):
            continue
        else:
            file_list.append(file)

        random.shuffle(file_list)
        random.shuffle(file_list)
        random.shuffle(file_list)
    print(file_list)

    data_num = len(file_list)

    for i in range(data_num):

        dir_name_img_src = os.path.join(SRC_PATH, file_list[i])
        dir_name_label_src = os.path.join(LABEL_PATH, file_list[i])
        if i > RATiO_VAL * data_num:
            dir_name_img = os.path.join(TRAIN_PATH, file_list[i])
            dir_name_label = os.path.join(TRAIN_LABEL_PATH, file_list[i])
        else:
            dir_name_img = os.path.join(VAL_PATH, file_list[i])
            dir_name_label = os.path.join(VAL_LABEL_PATH, file_list[i])

        open(dir_name_img, 'wb').write(open(dir_name_img_src, 'rb').read())
        open(dir_name_label, 'wb').write(open(dir_name_label_src, 'rb').read())


if __name__ == "__main__":
    main()
