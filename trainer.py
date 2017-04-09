from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, sys, os
# ignore CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
from data import input_data
from models import mnist

FLAGS = None


def main(_):
	#tf.logging.set_verbosity(tf.logging.INFO)

	estimator = tf.contrib.learn.Estimator(model_fn=mnist.model)
	print(estimator.fit(input_fn=input_data.input_fn, steps=1000))
	# print(estimator.evaluate(input_fn=input_data.input_fn, steps=10))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
