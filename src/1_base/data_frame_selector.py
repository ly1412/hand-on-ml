from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attrNames):
        self.attribute_names = attrNames
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values