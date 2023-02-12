from sklearn.datasets import fetch_openml
from joblib import Memory
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.linear_model import SGDClassifier


memory = Memory('./data')
fetch_openml_cached = memory.cache(fetch_openml)
mnist = fetch_openml_cached('mnist_784', data_home='data', cache=True)
print(mnist['DESCR'])
X, y = mnist['data'].values, mnist['target'].values
print(X.shape)
print(y.shape)
print(type(X))
some_digit = X[36000]
some_digit_img = some_digit.reshape(28, 28)
print(y[36000])
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
print('---------------------')
# print(np.unique(y_train))
# print(np.unique(y_test))
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
sgd_clf = SGDClassifier(random_state=42)
print('---------------------')
print(X_train.shape)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict(some_digit))