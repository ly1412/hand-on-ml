from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


iris = datasets.load_iris()
print(list(iris.keys()))
print(iris["data"].shape)
y = iris["target"]
# print(y.distinct())
X = iris["data"][:, (2, 3)] # petal length, petal width y = iris["target"]
print(X.shape)
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)
print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]))