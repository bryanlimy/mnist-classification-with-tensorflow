from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
from data.input_data import input_fn
from models import softmax, autoencoder

import tensorflow as tf
import os

# ignore CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'softmax', 'softmax, autoencoder or LSTM')
tf.app.flags.DEFINE_boolean('dropout', False, 'Use dropout or not')
tf.app.flags.DEFINE_float('dropout_prob', 0.5, 'Define dropout probability')


def accuracy_fn(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def euclidean_distance_fn(predictions, labels):
    return tf.reduce_mean(
        tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(predictions, labels)), 1))
    )


def main(_):
    model_fn = softmax.model_fn
    evaluation_metrics_fn = accuracy_fn
    evaluation_mode = 'accuracy'
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_buckets_scale = 0
    params = {'dropout': FLAGS.dropout, 'dropout_prob': FLAGS.dropout_prob}
    if FLAGS.model == 'autoencoder':
        model_fn = autoencoder.model_fn
        evaluation_mode = 'euclidean distance'
        evaluation_metrics_fn = euclidean_distance_fn

    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir='tmp/training',
        params=params,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=30)
    )

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=lambda: input_fn(data.train, 100, FLAGS.model,
                                        train_buckets_scale),
        eval_input_fn=lambda: input_fn(data.test, 100, FLAGS.model, train_buckets_scale),
        eval_metrics={evaluation_mode: evaluation_metrics_fn},
        train_steps=None,
        eval_steps=1,
        min_eval_frequency=1
    )

    experiment.train_and_evaluate()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
