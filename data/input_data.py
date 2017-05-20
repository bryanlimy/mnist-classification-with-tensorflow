from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def input_fn(data, batch_size, model, train_buckets_scale):
    if model == 'softmax':
        image, label = tf.train.slice_input_producer(
            [tf.constant(data.images), tf.constant(data.labels)])
        return tf.train.batch([image, label], batch_size=batch_size)
    elif model == 'autoencoder':
        image, label = tf.train.slice_input_producer(
            [tf.constant(data.images), tf.constant(data.labels)])
        return tf.train.batch([image, image], batch_size=batch_size)
