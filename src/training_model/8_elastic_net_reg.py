from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


m_ext = 2000
X = 6 * np.random.rand(m_ext, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m_ext, 1)
# Jθ=MSEθ+rα∑θi+ 2α∑θ2i
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
print(elastic_net.predict([[1.5]]))
