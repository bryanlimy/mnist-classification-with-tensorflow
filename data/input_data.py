import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def get_train_data():
	data = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.constant(data.train.images)
	y = tf.constant(data.train.labels)

	return x, y

def get_test_data():
	data = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.constant(data.test.images)
	y = tf.constant(data.test.labels)

	return x, y