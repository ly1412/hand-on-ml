from sklearn.tree import DecisionTreeRegressor
import pickle
import os
import sys


HOUSING_PATH = "datasets" + os.sep + "housing" + os.sep
read_file = os.path.join(sys.path[0], HOUSING_PATH + "tree_reg")
with open(read_file, 'rb') as pickle_file:
    tree_reg = pickle.load(pickle_file)
print(type(tree_reg))