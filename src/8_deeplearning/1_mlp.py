import tensorflow as tf
from joblib import Memory
from sklearn.datasets import fetch_openml


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

feature_columns = tf.compat.v1.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.compat.v1.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                             feature_columns=feature_columns)
dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=40000)
print(dnn_clf.evaluate(X_test, y_test))