import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def get_train_data():
	return input_fn(mnist.train, 100)

def get_test_data():
	return input_fn(mnist.test, 100)

def input_fn(input_data, batch_size):
	input_images = tf.constant(input_data.images)
	input_labels = tf.constant(input_data.labels)

	image, label = tf.train.slice_input_producer([input_images, input_labels])
	return tf.train.batch([image, label], batch_size=100)