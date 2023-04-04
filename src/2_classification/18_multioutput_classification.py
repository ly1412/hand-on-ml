from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.datasets import fetch_openml
from joblib import Memory
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import random as rnd


def plot_digit(some_digit):
    some_digit_img = some_digit.reshape(28, 28)
    plt.imshow(some_digit_img, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis('off')



memory = Memory('./data')
fetch_openml_cached = memory.cache(fetch_openml)
mnist = fetch_openml_cached('mnist_784', data_home='data', cache=True)
print(mnist['DESCR'])
X, y = mnist['data'].values, mnist['target'].values
print(X.shape)
print(y.shape)
print(type(X))
print(y[36000])
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
noise_train = np.random.randint(0, 100, (len(X_train), 784))
noise_test = np.random.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise_train
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test
# knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train_mod, y_train_mod)
plot_digit(X_test_mod[6666])
# clean_digit = knn_clf.predict([X_test_mod[6666]])
# plot_digit(clean_digit)
plt.show()
