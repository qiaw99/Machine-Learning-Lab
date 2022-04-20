""" sheet1_implementation.py

PUT YOUR NAMES HERE:
Qianli Wang
Feng Zhou


Write the functions
- pca
- gammaidx
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2021
"""
import numpy as np
import scipy.linalg as la


class PCA():
    def __init__(self, Xtrain):
        self.n, self.d = Xtrain.shape 
        
        # center the training data
        self.C = np.mean(Xtrain, axis=0)
        self.U = np.zeros((self.d, self.d))
        self.D = np.zeros((self.d, ))

        self.Xtrain = Xtrain - self.C
        cov = np.cov(self.Xtrain, rowvar=0)
        eigVals,self.eigVects=np.linalg.eig(np.mat(cov))
        self.eigVects = np.array(self.eigVects)
        
        self.U = self.eigVects
        self.D = np.array(eigVals)
        self.eigValIndice = np.argsort(eigVals)            #对特征值从小到大排序
        

    def project(self, Xtest, m):
        n_eigValIndice = self.eigValIndice[-1:-(m+1):-1]
        self.n_eigVect=self.eigVects[:,n_eigValIndice]
        self.lowDDataMat = self.Xtrain @ self.n_eigVect               #低维特征空间的数据
        return self.lowDDataMat

    def denoise(self, Xtest, m):
        return (self.lowDDataMat@self.n_eigVect.T)+ self.C  #重构数据


def gammaidx(X, k):
    res = []
    for i in range(X.shape[0]):
        _sum = 0
        distance = ((X[i] - X)**2).sum(axis=1) ** 0.5
        distance[i] = float('inf')
        distance = np.sort(distance)
        _sum = distance[:k].sum()
        res.append(_sum / k)
    return np.array(res)


def auc(y_true, y_pred, plot=False):
    # ...
    # you may use numpy.trapz here
    pass


def lle(X, m, n_rule, param, tol=1e-2):
    print('Step 1: Finding the nearest neighbours by rule ' + n_rule)
    # ...
    print('Step 2: local reconstruction weights')
    # ...
    print('Step 3: compute embedding')
    # ...
