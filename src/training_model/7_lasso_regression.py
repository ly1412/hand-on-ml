from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor


m_ext = 2000
X = 6 * np.random.rand(m_ext, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m_ext, 1)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
sgd_reg = SGDRegressor(penalty="l1")
sgd_reg.fit(X, y)
print("计算结果：", 0.5 * 1.5 ** 2 + 1.5 + 2)
print("lasso预测：", lasso_reg.predict([[1.5]]))
print("sgd l1预测：", sgd_reg.predict([[1.5]]))
