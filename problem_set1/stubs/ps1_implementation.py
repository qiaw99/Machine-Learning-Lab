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
from itertools import count
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
        # cov = np.cov(self.Xtrain, rowvar=0)

        N, M = self.Xtrain.shape
        cov = np.zeros((M, M))

        for i in range(M):
            mean_i = np.sum(self.Xtrain[:, i]) / N
            for j in range(M):
                mean_j = np.sum(self.Xtrain[:, j]) / N
                cov[i, j] = np.sum((self.Xtrain[:, i] - mean_i) * (self.Xtrain[:, j] - mean_j)) / (N - 1)

        eigVals,self.eigVects=np.linalg.eig(np.mat(cov))
        self.eigVects = np.array(self.eigVects)
        
        self.U = self.eigVects
        self.D = np.array(eigVals)
        # sort the eigen values asc
        self.eigValIndice = np.argsort(eigVals)    
        self.lowDDataMat = None        
        

    def project(self, Xtest, m):
        n_eigValIndice = self.eigValIndice[-1:-(m+1):-1]
        self.n_eigVect=self.eigVects[:,n_eigValIndice]

        # get data from lower dimension
        self.lowDDataMat = (Xtest-np.mean(Xtest, axis=0)) @ self.n_eigVect               
        return self.lowDDataMat

    def denoise(self, Xtest, m):
        # reconstruction
        if(self.lowDDataMat is None):
            return (self.lowDDataMat@self.n_eigVect.T) + np.mean(Xtest, axis=0)
        else:
            return (self.project(Xtest, m) @ self.n_eigVect.T) + np.mean(Xtest, axis=0)


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
    y_true = np.where(y_true==-1, 0, 1)
    data = np.concatenate((np.vstack(y_true[::-1]), np.vstack(y_pred[::-1])), axis=1)
    N = 120
    XY = np.zeros((N, 2))
    for i in range(N):
        y = i / 10
        params = xy(metrics(data, y))
        XY[i, 0] = params['fpr']
        XY[i, 1] = params['tpr']
    pre_x = None
    pre_y = None
    area = 0
    for i in range(N):
        x = XY[i, 0]
        y = XY[i, 1]
        if pre_x != x and pre_x is not None:
            area += (y + pre_y)/2 * (pre_x-x)
        pre_x = x
        pre_y = y
    return area


def metrics(data, y):
    predicts = data[:, 1] >= y
    # true pos
    tp = sum(data[predicts, 0])
    # false pos
    fp = data[predicts, 0].shape[0] - tp
    # false neg
    fn = sum(data[~predicts, 0])
    # true neg
    tn = data[~predicts, 0].shape[0] - fn
    return {
        'tp':tp,
        'fp':fp,
        'tn':tn,
        'fn':fn
    }   

def xy(params):
    fpr = params['fp'] / (params['fp']+params['tn'])
    tpr = params['tp'] / (params['tp']+params['fn'])
    return {
        'fpr':fpr,
        'tpr':tpr
    }

def lle(X, m, n_rule, tol=1e-2, k=None, epsilon=None):
    N = len(X)
    W = np.zeros([N,N])

    # print('Step 1: Finding the nearest neighbours by rule ' + n_rule)

    if(n_rule == "knn"):
        if k == None:
            raise ValueError("[Error] k cannot be None")
        for i in range(N):
            x_i = X[i]
            distance = np.sum(((x_i - X)**2), axis=1)
            # get k neigbors
            index = distance < (np.sort(distance))[k]
            np.delete(index, i)
            
            diff = x_i - X[index]
            C = diff @ diff.T + tol * np.identity(k)
            w = np.linalg.solve(C, np.ones(k))
            w /= w.sum()
            
            W[i, index] = w
   

    elif(n_rule =="eps-ball"):
        if epsilon == None:
            raise ValueError("[Error] epsilon cannot be None")
        for i in range(N):
            x_i = X[i]
            distance = np.sum(((x_i - X)**2), axis=1) ** 0.5
            distance[i] = float('inf')

            # filter all points that are inside epsilon ball
            index = distance < epsilon

            if(np.all(distance > epsilon) == True):
                raise ValueError("[Error] No neighbors")
            
            diff = x_i - X[index]
            k = diff.shape[0]

            C = diff @ diff.T + tol * np.identity(k)
            w = np.linalg.solve(C, np.ones(k))
            w /= w.sum()
            
            W[i, index] = w
    else:
        raise ValueError("[Error] Invalid n_rule")


    # print('Step 3: compute embedding')
    M = np.identity(N) - W - W.T + np.dot(W.T,W)
    
    # choose m embeddings
    E = np.linalg.svd(M)[0][:,-(1+m):-1]
    
    return E
    
