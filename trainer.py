from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
from data.input_data import input_fn
from models import softmax, autoencoder

import tensorflow as tf
import os, sys

# ignore CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'softmax', 'softmax, autoencoder or LSTM')
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
    """
    From tensorflow/models/tutorials/rnn/translate/translate.py read_data()
    
    Read data from source and target files and put into buckets.
  
    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).
  
    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(input_data.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


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
    if FLAGS.model == 'autoencoder':
        model_fn = autoencoder.model_fn
        evaluation_mode = 'euclidean distance'
        evaluation_metrics_fn = euclidean_distance_fn
    elif FLAGS.model == 'lstm':
        model_fn = None

        # From tensorflow/models/tutorials/rnn/translate/translate.py train()
        from_train, to_train, from_dev, to_dev, from_vocab, to_vocab \
            = input_data.prepare_wmt_data(
            'LSTM_data/',
            FLAGS.from_vocab_size,
            FLAGS.to_vocab_size)
        # Read data into buckets and compute their sizes.
        data.train = read_data(from_dev, to_dev)
        data.test = read_data(from_train, to_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(data.train[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir='tmp/training',
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
