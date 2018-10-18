import tensorflow as tf
import numpy as np
import os
from config import *
import cv2


def get_path(features_dir=DATASET_TRAIN, labels_dir=DATASET_TRAIN_LABEL):
    file_name = os.listdir(DATASET_TRAIN)
    features_list = []
    labels_list = []

    for item in file_name:
        features_list.append(os.path.join(features_dir, item))
        labels_list.append(os.path.join(labels_dir, item))
    return labels_list, features_list


def get_batch_data(features_dir=DATASET_TRAIN, labels_dir=DATASET_TRAIN_LABEL, batch_size=BATCH_SIZE_TRAIN, caps=64):
    labels, images = get_path(features_dir, labels_dir)
    images = tf.cast(images, tf.string)
    labels = tf.cast(labels, tf.string)
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
    image_path_batch, label_path_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=caps)
    ib = image_path_batch
    lb = label_path_batch
    return ib, lb


def generate_img_tensor():
    image_batch, label_batch = get_batch_data()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        i = 0
        try:
            while not coord.should_stop():
                image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
                i += 1
                img = []
                label = []
                for j in range(10):
                    img.append(cv2.imread(str(image_batch_v[j])[2:-1], 1))
                    label.append(cv2.imread(str(label_batch_v[j])[2:-1], cv2.IMREAD_GRAYSCALE))
                print(np.array(label).shape)
                return
        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    generate_img_tensor()
