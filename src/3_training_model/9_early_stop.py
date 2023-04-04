from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.base import clone
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


m_ext = 2000
X = 6 * np.random.rand(m_ext, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m_ext, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
scala = StandardScaler()
X_train_poly_scaled = scala.fit_transform(X_train)
X_val_poly_scaled = scala.fit_transform(X_val)
sgd_reg = SGDRegressor(n_iter_no_change=1, warm_start=True, penalty='l1', learning_rate="constant", eta0=0.0005)
minimum_val_error = float("inf")
best_epoch = None
# best_model = None

for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val_predict, y_val)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        # best_model = clone(sgd_reg)


print("Best epoch:", best_epoch)
print("Test data:", sgd_reg.predict([[1.5]]))