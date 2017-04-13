import tensorflow as tf

def model(features, targets, mode):
	# create model
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)

	predictions_dict = {"results": y}

	# define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	eval_metric_ops = {
		"rmse": tf.metrics.root_mean_squared_error(tf.cast(targets, tf.float32), y)
	}
	return tf.contrib.learn.ModelFnOps(
		mode=mode,
		predictions=predictions_dict,
		loss=cross_entropy,
		train_op=train_step,
		eval_metric_ops=eval_metric_ops)