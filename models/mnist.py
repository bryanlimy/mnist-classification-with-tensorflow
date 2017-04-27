import tensorflow as tf

def model(features, targets, mode):
	# create model
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	prediction = tf.nn.softmax(tf.matmul(features, W) + b)
	predictions_dict = {"results": prediction}

	# define loss and optimizer
	cross_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=prediction))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
		loss=cross_entropy,
		global_step=tf.contrib.framework.get_global_step())

	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(labels=targets, predictions=prediction, weights=None)
	}

	return tf.contrib.learn.ModelFnOps(
		mode=mode,
		predictions=predictions_dict,
		loss=cross_entropy,
		train_op=train_step,
		eval_metric_ops=eval_metric_ops)