import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def get_train_data():
	images, labels = mnist.train.next_batch(100)

	x = tf.constant(images)
	y = tf.constant(labels)

	return x, y

def get_test_data():
	images, labels = mnist.test.next_batch(100)

	x = tf.constant(images)
	y = tf.constant(labels)

	return x, y