import tensorflow as tf

def input_fn(data, batch_size):
	image, label = tf.train.slice_input_producer([tf.constant(data.images), tf.constant(data.labels)])
	return tf.train.batch([image, label], batch_size=batch_size)