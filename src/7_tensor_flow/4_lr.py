import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf


tf.compat.v1.disable_eager_execution()
housing = fetch_california_housing()
m, n = housing.data.shape
print('Housing shape:', housing.shape)
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)