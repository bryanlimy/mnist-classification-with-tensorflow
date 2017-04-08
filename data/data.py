# Download MINST data
from tensorflow.examples.tutorials.mnist import input_data

class Data:
	def __init__(self):
		# import MINST data
		self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	def get_trains_input(self):
		return self.mnist

	