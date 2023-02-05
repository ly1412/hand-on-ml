from sklearn.model_selection import StratifiedShuffleSplit
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


HOUSING_PATH = "housing"


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
housing = strat_train_set.copy()
# corelations pearson's r
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1)
plt.show()