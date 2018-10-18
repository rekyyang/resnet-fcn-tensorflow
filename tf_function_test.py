import tensorflow as tf
import numpy as np
import os
import cv2


def main():
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    image = tf.placeholder(tf.float32, [None, 2, 2])
    print(image.shape)
    bn = tf.layers.batch_normalization(image)
    mean, var = tf.nn.moments(image, axes=[0])

    image2 = tf.placeholder(tf.float32, [None, 256, 256, 3])
    conv1 = tf.layers.conv2d_transpose(image2,
                                       filters=6,
                                       kernel_size=[1, 1],
                                       # output_shape=[None, image2.shape[1]*2, image2.shape[2]*2, 6],
                                       strides=[2, 2],
                                       padding="SAME",
                                       name="conv1")
    print(conv1.shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r = sess.run(mean, feed_dict={image: a})
        print(a.shape[2])
    return 0


if __name__ == "__main__":
    filename = os.listdir("./dataset_train/train")
    filename_queue = tf.train.string_input_producer(filename, shuffle=True, num_epochs=MAX_EPOCHS)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)
    # main()

