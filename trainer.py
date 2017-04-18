from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, sys, os
# ignore CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from data import input_data
from models import mnist

FLAGS = None


def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)

	estimator = tf.contrib.learn.Estimator(model_fn=mnist.model)
	#estimator.fit(input_fn=input_data.get_train_data, steps=1000)
	#print(estimator.evaluate(input_fn=input_data.get_test_data, steps=10))

	experiment = tf.contrib.learn.Experiment(
		estimator=estimator,
		train_input_fn=input_data.get_train_data,
		eval_input_fn=input_data.get_test_data,
		eval_metrics=None,
		train_steps=1000,
		eval_steps=10, 
		train_monitors=None,
		eval_hooks=None,
		local_eval_frequency=None,
		eval_delay_secs=None,
		continuous_eval_throttle_secs=None,
		min_eval_frequency=None,
		delay_workers_by_global_step=None,
		export_strategies=None,
		continuous_eval_predicate_fn=None)

	print(experiment.train_and_evaluate())


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
