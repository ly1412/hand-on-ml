import tensorflow as tf
import numpy as np
from datetime import datetime


def relu(_x):
    w_shape = (int(_x.get_shape()[1]), 1)
    w = tf.Variable(tf.compat.v1.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(_x, w), b, name="z")
    return tf.maximum(z, 0., name="relu")


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
tf.compat.v1.disable_eager_execution()
n_features = 3
X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.compat.v1.add_n(relus, name="output")
output_summary = tf.compat.v1.summary.tensor_summary('output_summary', output)
file_writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    outputstr = output_summary.eval(feed_dict={X: np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])})
    file_writer.add_summary(outputstr)
file_writer.close()
