import numpy as np
import tensorflow as tf
import os as os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion
from custom_combined_attributes_adder import CustomCombinedAttributesAdder
from data_frame_selector import DataFrameSelector


HOUSING_PATH = "datasets/housing"
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    return pd.read_csv(csv_path)


tf.compat.v1.disable_eager_execution()
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
m, n = scaled_housing_data_plus_bias.shape
print('Housing shape:', scaled_housing_data_plus_bias.shape)
housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data_plus_bias]
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing_labels.to_numpy().reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.compat.v1.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.compat.v1.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)