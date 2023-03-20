from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor


m_ext = 2000
X = 6 * np.random.rand(m_ext, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m_ext, 1)
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
print(ridge_reg.predict([[1.5]]))
print(0.5 * 1.5 ** 2 + 1.5 + 2)

sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))