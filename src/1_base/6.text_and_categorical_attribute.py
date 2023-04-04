import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


HOUSING_PATH = "datasets/housing"


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    return pd.read_csv(csv_path)


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
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
print("Train set shape:", strat_train_set.shape)
print("Test set shape:", strat_test_set.shape)
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()
# housing_cat_onehot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print(type(housing_cat_onehot))
# print(housing_cat_onehot.toarray())
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer(sparse_output=True)
housing_cat_onehot = encoder.fit_transform(housing_cat)
print(housing_cat_onehot)