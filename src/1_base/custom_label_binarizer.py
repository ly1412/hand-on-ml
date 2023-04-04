from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelBinarizer


class CustomLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=None):
        return self.encoder.transform(x)