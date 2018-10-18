import json
import os
import cv2
import numpy as np


SRC_PATH = './dataset/labels/'

DST_PATH = './dataset/label_images/'

json_list = []

img_list = []


def main():
    for file in os.listdir(SRC_PATH):
        if os.path.isdir(file):
            continue
        else:
            file = os.path.join(SRC_PATH, file)
            json_list.append(file)

    for file in json_list:
        fp = open(file)
        json_data = json.load(fp=fp)
        img = cv2.imread(os.path.join(SRC_PATH, json_data["imagePath"]).replace(' ', ''))
        dst_path = os.path.join(DST_PATH, json_data["imagePath"].replace(' ', '').replace("..", '.'))
        height = img.shape[0]
        width = img.shape[1]
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        for shape in json_data["shapes"]:
            pts = []
            for point in shape["points"]:
                pts.append(point)
            pts = np.array(pts)
            pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (255, 255, 255))
        cv2.imwrite(dst_path, mask)


if __name__ == "__main__":
    main()
