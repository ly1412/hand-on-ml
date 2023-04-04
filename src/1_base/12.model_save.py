from sklearn.model_selection import cross_val_score
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from custom_combined_attributes_adder import CustomCombinedAttributesAdder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from data_frame_selector import DataFrameSelector
from custom_label_binarizer import CustomLabelBinarizer
from sklearn.metrics import mean_squared_error
import sys
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pickle

HOUSING_PATH = "datasets" + os.sep + "housing" + os.sep


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(sys.path[0], HOUSING_PATH + "housing.csv")
    return pd.read_csv(csv_path)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


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

cat_pipline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_banarizer', CustomLabelBinarizer())
])

full_pipe = FeatureUnion(transformer_list=[
    ('numbs', num_pipeline),
    ('categorical', cat_pipline),
])

housing_prepared = full_pipe.fit_transform(housing)
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)

# rnd_forest = RandomForestRegressor()
# rnd_forest.fit(housing_prepared, housing_labels)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)
save_file = os.path.join(sys.path[0], HOUSING_PATH + "tree_reg")
print("OutputFile:", save_file)
with open(save_file, 'wb') as pickle_file:
    pickle.dump(tree_reg, pickle_file)

