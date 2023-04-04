from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


iris = datasets.load_iris()
print(list(iris.keys()))
print(iris["data"].shape)
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(int)
print("y shape:", y.shape)
log_reg = LogisticRegression()
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(X_new.shape)
y_proba = log_reg.predict_proba(X_new)
print(y_proba.shape)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
plt.legend()
plt.show()

# The hyperparameter controlling the regularization strength of a Scikit-Learn LogisticRegression model is not alpha (as in other linear models), but its inverse: C. The higher the value of C, the less the model is regularized.