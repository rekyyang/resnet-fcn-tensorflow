import cv2
import os

SRC_PATH = '../dataset_orig/label_images/'  # /labels/'
DST_PATH = '../dataset/labels/'


def main():
    for file in os.listdir(SRC_PATH):
        if os.path.isdir(os.path.join(SRC_PATH, file)):
            continue
        else:
            img = cv2.imread(os.path.join(SRC_PATH, file))
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(DST_PATH, file), img)


if __name__ == "__main__":
    main()
