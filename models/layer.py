import tensorflow as tf

class Model:
	def __init__(self):
		# create model
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.W = tf.Variable(tf.zeros([784, 10]))
		self.b = tf.Variable(tf.zeros([10]))
		self.y = tf.nn.softmax(tf.matmul(x, W) + b)

		# define loss and optimizer
		self.y_ = tf.placeholder(tf.float32, [None, 10])
		self.cross_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

		self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	def trainModel(self, mnist):
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		# train model 1000 times
		for _ in range(1000):
			batch_xs, batch_ys = mnist.train.next_batch(100)
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

		return tf.argmax(y, 1)