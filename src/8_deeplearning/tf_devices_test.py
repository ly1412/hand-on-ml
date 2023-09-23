import tensorflow as tf


sess = tf.compat.v1.Session()
print(sess)

with tf.device("/cpu:0"):
    a = tf.compat.v1.Variable(3.0)
    b = tf.compat.v1.constant(4.0)

c = a*b