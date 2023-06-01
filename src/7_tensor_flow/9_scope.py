import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion
from custom_combined_attributes_adder import CustomCombinedAttributesAdder
from data_frame_selector import DataFrameSelector
from datetime import datetime


HOUSING_PATH = "datasets/housing"


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    return pd.read_csv(csv_path)


def fetch_batch(_epoch, batch_index, _batch_size, _scaled_housing_data_plus_bias, _label):
    end_ind = (batch_index + 1) * _batch_size if ((batch_index + 1) * _batch_size < len(
        _scaled_housing_data_plus_bias)) else len(_scaled_housing_data_plus_bias)
    _X_batch, _y_batch = _scaled_housing_data_plus_bias[batch_index*_batch_size:end_ind],\
        _label[batch_index * _batch_size : end_ind]
    return _X_batch, _y_batch


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
housing = load_housing_data()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# 按重要属性分层并进行split
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# 小于5的不处理，大于5的用5.0填充
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# 删除列
for set_ in (strat_train_set, strat_test_set):
    set_.drop(["income_cat"], axis=1, inplace=True)
print("Train set shape:", strat_train_set.shape)
print("Test set shape:", strat_test_set.shape)
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
housing_num = housing.drop('ocean_proximity', axis=1)
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CustomCombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

scaled_housing_data_plus_bias = num_pipeline.fit_transform(housing)
print("scaled shape:", scaled_housing_data_plus_bias.shape)
m, n = scaled_housing_data_plus_bias.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data_plus_bias]
print("add bais shape:", housing_data_plus_bias.shape)
tf.compat.v1.disable_eager_execution()
n_epochs = 100
learning_rate = 0.001
X = tf.compat.v1.placeholder(tf.float32, shape=(None, n), name="X")
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="y")
batch_size = 500
n_batches = int(np.ceil(m / batch_size))
theta = tf.compat.v1.Variable(tf.compat.v1.random_uniform([n, 1], -1.0, 1.0), name="theta")
y_pred = tf.compat.v1.matmul(X, theta, name="predictions")
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
saver = tf.compat.v1.train.Saver()
mse_summary = tf.compat.v1.summary.scalar('MSE', mse)
file_writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_ind in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_ind, batch_size, scaled_housing_data_plus_bias, housing_labels)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch.to_numpy().reshape(-1, 1)})
            if batch_ind % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch.to_numpy().reshape(-1, 1)})
                step = epoch * n_batches + batch_ind
                file_writer.add_summary(summary_str, step)
        if n_epochs % 5 == 0:
            save_path = saver.save(sess, 'D:\\tmp\\model_data\\my_model.ckpt')
    best_theta = theta.eval()
file_writer.close()
print(best_theta)
