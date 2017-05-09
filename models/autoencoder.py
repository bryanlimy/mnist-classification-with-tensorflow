import tensorflow as tf


def autoencoder_fn(features, targets, mode):
    n_hidden_1 = 50    # 1st layer number of features
    n_hidden_2 = 50    # 2nd layer number of features
    n_input = 784       # MNIST data input

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input]))
    }

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        return layer_2

    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        return layer_2

    tf.summary.image("input", tf.reshape(features, [-1, 28, 28, 1]))

    encoder_op = encoder(features)
    decoder_op = decoder(encoder_op)

    y_prediction = decoder_op
    y_true = features

    tf.summary.image("output", tf.reshape(y_prediction, [-1, 28, 28, 1]))

    cost = tf.reduce_mean(tf.pow(y_true - y_prediction, 2))

    train_step = tf.train.AdamOptimizer(0.001).minimize(
        loss=cost,
        global_step=tf.contrib.framework.get_global_step()
    )

    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions=y_prediction,
        loss=cost,
        train_op=train_step
    )
