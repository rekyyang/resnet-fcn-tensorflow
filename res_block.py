import tensorflow as tf
import numpy as np
import cv2
from config import *
import time
import os
import matplotlib.pyplot as plt
import glob
import random
from read_data import *


# net structure
class ResFcn:
	def __init__(self):
		self.input_img_batch = tf.placeholder(tf.float32, [None, 512, 512, 3], name='image_input')
		self.labels = tf.placeholder(tf.float32, [None, 512, 512, 1], name='labels')
		self.label_1 = self.labels / 256
		self.seg = self.resnet(self.input_img_batch)
		self.loss = tf.reduce_mean(tf.square(self.seg - self.label_1))
		# self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.seg, labels=self.labels))
		self.train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)
		self.saver = tf.train.Saver()

	def input_block(self, img, out_channels=64, kernel_size=7, strides=2):
		with tf.variable_scope("input"):
			conv = tf.layers.conv2d(img,
									filters=out_channels,
									kernel_size=[kernel_size, kernel_size],
									strides=[strides, strides],
									padding="SAME",
									name="conv")
			return conv

	def res_block_base(self, img, out_channels, name, kernel_size=3, strides=1):
		with tf.variable_scope(name):
			bn1 = tf.layers.batch_normalization(img, momentum=0.4, name="bn1")
			relu1 = tf.nn.elu(bn1)
			conv1 = tf.layers.conv2d(relu1,
									 filters=out_channels,
									 kernel_size=[kernel_size, kernel_size],
									 strides=[strides, strides],
									 padding="SAME",
									 name="conv1")

			bn2 = tf.layers.batch_normalization(conv1, momentum=0.4, name="bn2")
			relu2 = tf.nn.elu(bn2)
			conv2 = tf.layers.conv2d(relu2,
									 filters=out_channels,
									 kernel_size=[kernel_size, kernel_size],
									 strides=[strides, strides],
									 padding="SAME",
									 name="conv2")
			result = tf.add(conv2, img)
			return result

	def res_block_bottleneck_down(self, img, out_channels, name, kernel_size=3, strides=1):
		with tf.variable_scope(name):
			bn1 = tf.layers.batch_normalization(img, momentum=0.4, name="bn1")
			relu1 = tf.nn.elu(bn1)
			conv1 = tf.layers.conv2d(relu1,
									 filters=out_channels,
									 kernel_size=[1, 1],
									 strides=[2, 2],
									 padding="SAME",
									 name="conv1_down")

			bn2 = tf.layers.batch_normalization(conv1, momentum=0.4, name="bn2")
			relu2 = tf.nn.elu(bn2)
			conv2 = tf.layers.conv2d(relu2,
									 filters=out_channels,
									 kernel_size=[kernel_size, kernel_size],
									 strides=[strides, strides],
									 padding="SAME",
									 name="conv2")

			bn3 = tf.layers.batch_normalization(conv2, momentum=0.4, name="bn3")
			relu3 = tf.nn.elu(bn3)
			conv3 = tf.layers.conv2d(relu3,
									 filters=out_channels,
									 kernel_size=[kernel_size, kernel_size],
									 strides=[strides, strides],
									 padding="SAME",
									 name="conv3")
			result = tf.add(conv3, conv1)
			return result

	def res_block_bottleneck_up(self, img, out_channels, name, kernel_size=3, strides=1):
		with tf.variable_scope(name):
			bn1 = tf.layers.batch_normalization(img, momentum=0.4, name="bn1")
			relu1 = tf.nn.elu(bn1)
			conv1 = tf.layers.conv2d_transpose(relu1,
												filters=out_channels,
											    kernel_size=[1, 1],
												# output_shape=[None, relu1.shape[1]*2, relu1.shape[2]*2, out_channels],
											    strides=[2, 2],
											    padding="SAME",
											    name="conv1_up")

			bn2 = tf.layers.batch_normalization(conv1, momentum=0.4, name="bn2")
			relu2 = tf.nn.elu(bn2)
			conv2 = tf.layers.conv2d(relu2,
									 filters=out_channels,
									 kernel_size=[kernel_size, kernel_size],
									 strides=[strides, strides],
									 padding="SAME",
									 name="conv2")

			bn3 = tf.layers.batch_normalization(conv2, momentum=0.4, name="bn3")
			relu3 = tf.nn.elu(bn3)
			conv3 = tf.layers.conv2d(relu3,
									 filters=out_channels,
									 kernel_size=[kernel_size, kernel_size],
									 strides=[strides, strides],
									 padding="SAME",
									 name="conv3")
			result = tf.add(conv3, conv1)
			return result

	# def res_block_3(self, img, out_channels, down=True):
	# 	with tf.variable_scope("res_" + str(img.shape[3]) + str("down" if down else "up")):
	# 		res1 = self.res_block_base(img, img.shape[3], "res_block1")
	# 		res2 = self.res_block_base(res1, res1.shape[3], "res_block2")
	# 		if down:
	# 			res3 = self.res_block_bottleneck_down(res2, out_channels, "res_block3")
	# 		else:
	# 			res3 = self.res_block_bottleneck_up(res2, out_channels, "res_block3")
	# 		return res3

	def res_block_3(self, img, out_channels, down=True):
		with tf.variable_scope("res_" + str(img.shape[3]) + 'to' + str(out_channels) + str("down" if down else "up")):
			res1 = self.res_block_base(img, img.shape[3], "res_block1")
			res2 = self.res_block_base(res1, res1.shape[3], "res_block2")
			if down:
				res3 = self.res_block_bottleneck_down(res2, out_channels, "res_block3")
			else:
				res3 = self.res_block_bottleneck_up(res2, out_channels, "res_block3")
			return res3

	def resnet(self, img):

		img_ = self.input_block(img)

		res_blk_64 = self.res_block_3(img_, 128)

		res_blk_128 = self.res_block_3(res_blk_64, 256)

		res_blk_256 = self.res_block_3(res_blk_128, 512)
		# ------------------------------------------------------------------------------------------------------------ #
		res_blk_512_ = self.res_block_3(res_blk_256, 256, False)
		# res_blk_512_ = res_blk_128

		res_blk_256_ = self.res_block_3(tf.concat([res_blk_512_, res_blk_128], 3), 128, False)

		res_blk_128_ = self.res_block_3(tf.concat([res_blk_256_, res_blk_64], 3), 64, False)

		res_blk_64_ = self.res_block_3(tf.concat([res_blk_128_, img_], 3), 1, False)

		result = res_blk_64_

		print(result)

		return result

	def train(self):
		image_batch, label_batch = get_batch_data()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.saver.restore(sess, tf.train.latest_checkpoint("./weight"))
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess, coord)
			i = 0
			try:
				ii = 0
				plt.figure(0)
				while not coord.should_stop():
					image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
					i += 1
					img = []
					label = []
					for j in range(BATCH_SIZE_TRAIN):
						img.append(cv2.imread(str(image_batch_v[j])[2:-1], 1))
						label.append(cv2.imread(str(label_batch_v[j])[2:-1], cv2.IMREAD_GRAYSCALE))
					label = np.array(label)
					label = label[:, :, :, np.newaxis]
					loss = sess.run(self.loss, feed_dict={self.input_img_batch: img,
														  self.labels: label})
					print(loss)
					sess.run(self.train_op, feed_dict={self.input_img_batch: img,
													   self.labels: label})

					# img_ = cv2.imread('./dataset/test/(22).jpg', 1)
					# img_ = cv2.resize(img_, (512, 512))
					# img_ = np.array(img_)[np.newaxis, :, :, :]
					# feed_dict = {self.input_img_batch: img_}
					# result = sess.run(self.seg, feed_dict=feed_dict)
					# plt.imshow(np.array(result[0, :, :, 0]))
					# plt.show()

					if ii >= 1000 and ii % 100 == 0:
						self.saver.save(sess, MODEL_PATH)
					ii = ii + 1
					if ii > 20000:
						break
			except tf.errors.OutOfRangeError:
				print("done")
			finally:
				coord.request_stop()
			coord.join(threads)

	def predict(self, img_in):
		img = cv2.imread(img_in, 1)
		img = cv2.resize(img, (512, 512))
		img = np.array(img)[np.newaxis, :, :, :]
		with tf.Session() as sess:
			self.saver = tf.train.import_meta_graph("./weight/model.meta")
			self.saver.restore(sess, tf.train.latest_checkpoint("./weight"))
			graph = tf.get_default_graph()
			result_tensor = graph.get_tensor_by_name("res_128to1up/res_block3/Add:0")
			input_tensor = graph.get_tensor_by_name("image_input:0")
			feed_dict = {input_tensor: img}
			result = sess.run(result_tensor, feed_dict=feed_dict)
			print(result)
			plt.figure(0)
			plt.imshow(np.array(result[0, :, :, 0]))
			plt.show()


if __name__ == "__main__":
	res_fcn = ResFcn()
	res_fcn.predict('./dataset/test/3.jpg')
	# res_fcn.train()

