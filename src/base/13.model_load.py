from sklearn.tree import DecisionTreeRegressor
import pickle
import os
import sys


HOUSING_PATH = "datasets" + os.sep + "housing" + os.sep
with open(os.path.join(sys.path[0], HOUSING_PATH + "tree_reg"), 'rb') as pickle_file:
    tree_reg = pickle.load(pickle_file)
print(type(tree_reg))