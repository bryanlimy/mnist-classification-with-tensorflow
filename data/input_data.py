# Download MINST data
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

def input_fn():
	data = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.constant(data.train.images)
	y = tf.constant(data.train.labels)

	return x, y
