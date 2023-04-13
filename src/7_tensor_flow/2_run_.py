import tensorflow as tf


tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.Variable(3.0, name="x")
y = tf.compat.v1.Variable(4.0, name="y")
f = x*x*y + y + 2
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    init.run()
    result = f.eval()
    print(result)
