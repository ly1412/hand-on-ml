import tensorflow as tf


tf.compat.v1.disable_eager_execution()
A = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))
B = A + 5

with tf.compat.v1.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
print(B_val_1)
print(B_val_2)
