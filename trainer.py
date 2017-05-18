from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
from data.input_data import input_fn
from models.default import default_fn
from models.autoencoder import autoencoder_fn

import tensorflow as tf
import os
# ignore CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'default', 'default or auto-encoder model')


def accuracy_fn(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def euclidean_distance_fn(predictions, labels):
    return tf.reduce_mean(
        tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predictions, labels)), 1))
    )

def main(_):
    model_fn = default_fn
    evaluation_metrics_fn = accuracy_fn
    evaluation_mode = 'accuracy'
    if FLAGS.model == 'autoencoder':
        model_fn = autoencoder_fn
        evaluation_mode = 'euclidean distance'
        evaluation_metrics_fn = euclidean_distance_fn

    data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    estimator = tf.contrib.learn.Estimator(
		model_fn=model_fn,
		model_dir='tmp/training',
		config=tf.contrib.learn.RunConfig(save_checkpoints_secs=30)
	)

    experiment = tf.contrib.learn.Experiment(
		estimator=estimator,
		train_input_fn=lambda: input_fn(data.train, 100, FLAGS.model),
		eval_input_fn=lambda: input_fn(data.test, 100, FLAGS.model),
		eval_metrics={evaluation_mode: evaluation_metrics_fn},
		train_steps=None,
		eval_steps=1,
		min_eval_frequency=1
	)

    experiment.train_and_evaluate()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
