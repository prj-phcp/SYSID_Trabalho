from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
import numpy as np


class DimensionMismatchException(Exception):
    
    pass


class SYSIDRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    '''
    Classe para identificacao de sistemas em conjunto com o sklearn.
    Herda dos principais objetos do sklearn
    '''

    def __init__(self, nX, ny, preprocessor=None, expansor=None, estimator=LinearRegression(fit_intercept=False)):

        self.nX = nX
        self.ny = ny
        self.preprocessor = preprocessor
        self.expansor = expansor
        self.estimator = estimator
        

    def fit(self, X, y):
        X, y = self.fit_transform(X, y)
        #if len(y.shape) == 2 and y.shape[1] == 1:
        #    y.reshape(-1)
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y, steps_ahead=None):

        if steps_ahead == 1:
            yhat = self.__OSA_predict(X, y)
        elif steps_ahead is None:
            yhat = self.__FS_predict(X, y)
        else:
            yhat = self.__nS_predict(X, y, steps_ahead)
        return yhat


    def fit_predict(self, X, y, steps_ahead=None):
        return self.fit(X, y).predict(X, y, steps_ahead)


    def transform(self, X, y):
        if not self.preprocessor is None:
            X = self.preprocessor.transform(X)
        y, X = self.__matReg(X, y)
        if not self.expansor is None:
            X = self.expansor.transform(X)
        return X, y


    def fit_transform(self, X, y):
        if not self.preprocessor is None:
            X = self.preprocessor.fit_transform(X)
        y, X = self.__matReg(X, y)
        if not self.expansor is None:
            X = self.expansor.fit_transform(X)
        return X, y


    def __estimate_parameters(self, X, y):

        nx = self.nX
        ny = self.ny

        if len(X.shape) == 1: X = X.reshape(-1,1)
        if len(y.shape) == 1: y = y.reshape(-1,1)

        (N, ky) = y.shape
        (Nx,kx) = X.shape

        if isinstance(nx, int):
            nx = [nx]*kx
        if isinstance(ny, int):
            ny = [ny]*ky

        p = np.max(ny+nx) + 1

        if N != Nx:
            raise DimensionMismatchException(f'Dimensions of input and output are mismatched. Dimensions of X are {X.shape} and y are {y.shape}')

        return nx, ny, kx, ky, N, Nx, p


    def __matReg(self,X,Y):
        
        nx, ny, kx, ky, N, Nx, p = self.__estimate_parameters(X, Y)
            
        # create target vector
        target = Y[p-1:,:]

        # create regression matrix
        Phi = np.zeros((N-p+1,np.sum(ny+nx)))
        counter = 0
        for i in range(ky):
            ny_i = ny[i]
            y = Y[:,i]
            for j in range(ny_i):
                Phi[:,counter] = y[p-j-2: N-j-1].reshape(-1)
                counter +=1
                #print(Phi.shape, counter)

        for i in range(kx):
            nx_i = nx[i]
            x = X[:,i]
            for j in range(nx_i):
                Phi[:,counter] = x[p-j-2: N-j-1].reshape(-1)
                counter +=1
                #print(Phi.shape, counter)

        return (target, Phi)

    def __OSA_predict(self, X, y):

        nx, ny, kx, ky, N, Nx, p = self.__estimate_parameters(X, y)
        y = y.reshape(-1,ky)
        yhat = np.zeros(y.shape)
        yhat[:p-1,:] = y[:p-1,:]
        X, y = self.transform(X, y)
        yhat[p-1:,:] = self.estimator.predict(X).reshape(-1,ky)
        return yhat

    def __nS_predict(self, X, y, steps_ahead):

        nx, ny, kx, ky, N, Nx, p = self.__estimate_parameters(X, y)
        y = y.reshape(-1,ky)
        yhat = np.zeros(y.shape)
        yhat[:p-1,:] = y[:p-1,:]
        Phi, _ = self.transform(X, y)
        Phi_i = Phi[0,:].reshape(1,-1)
        for i in range(p-1,N):
            yhat[i,:] = self.estimator.predict(Phi_i).reshape(-1)
            if (1+i-p) % steps_ahead != 0:
                Phi, _ = self.transform(X[:i+1,:], yhat[:i+1,:])
            else:
                Phi, _ = self.transform(X[:i+1,:], y[:i+1,:])
            Phi_i = Phi[-1,:].reshape(1,-1)
        return yhat

    def __FS_predict(self, X, y):

        nx, ny, kx, ky, N, Nx, p = self.__estimate_parameters(X, y)
        y = y.reshape(-1,ky)
        yhat = np.zeros(y.shape)
        yhat[:p-1,:] = y[:p-1,:]
        Phi, y = self.transform(X, y)
        Phi_i = Phi[0,:].reshape(1,-1)
        for i in range(p-1,N):
            yhat[i,:] = self.estimator.predict(Phi_i).reshape(-1)
            Phi, _ = self.transform(X[:i+1,:], yhat[:i+1,:])
            Phi_i = Phi[-1,:].reshape(1,-1)
        return yhat


class CONTSIDEstimator(BaseEstimator, RegressorMixin, TransformerMixin):
    '''
    Classe definida para a aplicação de derivadas contínuas a um sinal com espaçamento regular.
    '''

    pass



