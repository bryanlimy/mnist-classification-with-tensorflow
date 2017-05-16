import tensorflow as tf


def autoencoder_fn(features, targets, mode):
    n_hidden = 50    # layer number of features
    n_input = 784       # MNIST data input

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'encoder_h3': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'encoder_h4': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'encoder_h5': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'encoder_h6': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'encoder_h7': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'encoder_h8': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'encoder_h9': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'encoder_h10': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'decoder_h3': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'decoder_h4': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'decoder_h5': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'decoder_h6': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'decoder_h7': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'decoder_h8': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'decoder_h9': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
        'decoder_h10': tf.Variable(tf.random_normal([n_hidden, n_input]))
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden])),
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden])),
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden])),
        'encoder_b5': tf.Variable(tf.random_normal([n_hidden])),
        'encoder_b6': tf.Variable(tf.random_normal([n_hidden])),
        'encoder_b7': tf.Variable(tf.random_normal([n_hidden])),
        'encoder_b8': tf.Variable(tf.random_normal([n_hidden])),
        'encoder_b9': tf.Variable(tf.random_normal([n_hidden])),
        'encoder_b10': tf.Variable(tf.random_normal([n_hidden])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden])),
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden])),
        'decoder_b4': tf.Variable(tf.random_normal([n_hidden])),
        'decoder_b5': tf.Variable(tf.random_normal([n_hidden])),
        'decoder_b6': tf.Variable(tf.random_normal([n_hidden])),
        'decoder_b7': tf.Variable(tf.random_normal([n_hidden])),
        'decoder_b8': tf.Variable(tf.random_normal([n_hidden])),
        'decoder_b9': tf.Variable(tf.random_normal([n_hidden])),
        'decoder_b10': tf.Variable(tf.random_normal([n_input]))
    }

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                       biases['encoder_b3']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                       biases['encoder_b4']))
        layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['encoder_h5']),
                                       biases['encoder_b5']))
        layer_6 = tf.nn.sigmoid(tf.add(tf.matmul(layer_5, weights['encoder_h6']),
                                       biases['encoder_b6']))
        layer_7 = tf.nn.sigmoid(tf.add(tf.matmul(layer_6, weights['encoder_h7']),
                                       biases['encoder_b7']))
        layer_8 = tf.nn.sigmoid(tf.add(tf.matmul(layer_7, weights['encoder_h8']),
                                       biases['encoder_b8']))
        layer_9 = tf.nn.sigmoid(tf.add(tf.matmul(layer_8, weights['encoder_h9']),
                                       biases['encoder_b9']))
        layer_10 = tf.nn.sigmoid(tf.add(tf.matmul(layer_9, weights['encoder_h10']),
                                       biases['encoder_b10']))
        return layer_10

    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                       biases['decoder_b3']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                       biases['decoder_b4']))
        layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h5']),
                                       biases['decoder_b5']))
        layer_6 = tf.nn.sigmoid(tf.add(tf.matmul(layer_5, weights['decoder_h6']),
                                       biases['decoder_b6']))
        layer_7 = tf.nn.sigmoid(tf.add(tf.matmul(layer_6, weights['decoder_h7']),
                                       biases['decoder_b7']))
        layer_8 = tf.nn.sigmoid(tf.add(tf.matmul(layer_7, weights['decoder_h8']),
                                       biases['decoder_b8']))
        layer_9 = tf.nn.sigmoid(tf.add(tf.matmul(layer_8, weights['decoder_h9']),
                                       biases['decoder_b9']))
        layer_10 = tf.nn.sigmoid(tf.add(tf.matmul(layer_9, weights['decoder_h10']),
                                       biases['decoder_b10']))
        return layer_10

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
