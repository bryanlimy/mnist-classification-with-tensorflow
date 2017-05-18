import tensorflow as tf

def input_fn(data, batch_size, model):
    image, label = tf.train.slice_input_producer([tf.constant(data.images), tf.constant(data.labels)])
    if model == 'autoencoder':
        return tf.train.batch([image, image], batch_size=batch_size)
    else:
        return tf.train.batch([image, label], batch_size=batch_size)
