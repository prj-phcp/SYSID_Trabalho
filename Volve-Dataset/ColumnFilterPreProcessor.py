from scipy import stats
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SingleColumnFilter(BaseEstimator, TransformerMixin):

    def __init__(self, column_index):

        self.column_index = column_index

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[:,self.column_index].reshape(-1,1)

    def fit_transform(self, X, y=None):

        return self.fit(X, y).transform(X)

class QuantileColumnFilter(BaseEstimator, TransformerMixin):

    def __init__(self, random_state=None, predictor=stats.uniform(), quantile=0.5):

        self.random_state = random_state
        self.predictor = predictor
        self.quantile = quantile
        self.__filtlist = None

        return self

    def fit(self, X, y=None):
        
        if len(X.shape) == 1: X.reshape(-1,1)
        ( _ , nX) = X.shape
        self.__fitlist = [0]
        counter = 1
        while np.sum(self.__fitlist) == 0:
            self.__fitlist = (self.predictor.rvs(counter*nX, random_state = self.random_state) <= self.quantile)[(1-counter)*nX:]

    def transform(self, X):

        return X[:,self.__fitlist]

        
    def fit_transform(self, X, y=None):

        return self.fit(X, y).transform(X)