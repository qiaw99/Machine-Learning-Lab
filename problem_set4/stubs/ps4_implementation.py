""" ps4_implementation.py

PUT YOUR NAME HERE:
Qianli Wang and Feng Zhou


Complete the classes and functions
- svm_qp
- plot_svm_2d
- neural_network
Write your implementations in the given functions stubs!


(c) Felix Brockherde, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2019
"""
import imp
import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch

class svm_qp():
    """ Support Vector Machines via Quadratic Programming """

    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None
    
    def fit(self, X, Y):

        # INSERT_CODE
        
        # Here you have to set the matrices as in the general QP problem
        #P = 
        #q = 
        #G = 
        #h = 
        #A =   # hint: this has to be a row vector
        #b =   # hint: this has to be a scalar
        
        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()

        #b = 

    def predict(self, X):

        # INSERT_CODE

        return self


# This is already implemented for your convenience
class svm_sklearn():
    """ SVM via scikit-learn """
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'
        self.clf = sklearn.svm.SVC(C=C,
                                   kernel=kernel,
                                   gamma=1./(1./2. * kernelparameter ** 2),
                                   degree=kernelparameter,
                                   coef0=kernelparameter)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.X_sv = X[self.clf.support_, :]
        self.y_sv = y[self.clf.support_]

    def predict(self, X):
        return self.clf.decision_function(X)


def plot_boundary_2d(X, y, model):
    X_pos = X[np.where(y > 0)]
    X_neg = X[np.where(y <= 0)]
    maxi = np.max(X, axis=0)
    mini = np.min(X, axis=0)
    xm = np.arange(mini[0], maxi[0], 0.01)
    ym = np.arange(mini[1], maxi[1], 0.01)
    xx, yy = np.meshgrid(xm, ym, indexing='xy')
    xy = np.array([xx, yy])
    xy = xy.T
    coordinates = xy.reshape((-1, 2))
    preds = model.predict(coordinates)
    if preds.ndim > 1:
        preds = (-preds).argmax(1)
    preds = preds.reshape(len(xm), len(ym)) > 0
    plt.figure(figsize=(15, 10))
    plt.contourf(xm, ym, preds.T)
    if hasattr(model, 'X_sv'):
        plt.scatter(model.X_sv.T[0], model.X_sv.T[1], marker='+', c='r', s=300)
    plt.scatter(X_pos.T[0], X_pos.T[1])
    plt.scatter(X_neg.T[0], X_neg.T[1])
    plt.show()


def sqdistmat(X, Y=False):
    if Y is False:
        X2 = sum(X**2, 0)[np.newaxis, :]
        D2 = X2 + X2.T - 2*np.dot(X.T, X)
    else:
        X2 = sum(X**2, 0)[:, np.newaxis]
        Y2 = sum(Y**2, 0)[np.newaxis, :]
        D2 = X2 + Y2 - 2*np.dot(X.T, Y)
    return D2


def buildKernel(X, Y=False, kernel='linear', kernelparameter=0):
    d, n = X.shape
    if Y.isinstance(bool) and Y is False:
        Y = X
    if kernel == 'linear':
        K = np.dot(X.T, Y)
    elif kernel == 'polynomial':
        K = np.dot(X.T, Y) + 1
        K = K**kernelparameter
    elif kernel == 'gaussian':
        K = sqdistmat(X, Y)
        K = np.exp(K / (-2 * kernelparameter**2))
    else:
        raise Exception('unspecified kernel')
    return K

from torch import nn

class neural_network(nn.Module):
    def __init__(self, layers=[2,100,2], scale=.1, p=None, lr=None, lam=None):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(scale*torch.randn(m, n)) for m, n in zip(layers[:-1], layers[1:])])
        self.biases = nn.ParameterList([nn.Parameter(scale*torch.randn(n)) for n in layers[1:]])

        self.p = p
        self.lr = lr
        self.lam = lam
        self.train = False

    def relu(self, X, W, b):
        rng = np.random.RandomState()
        mask = rng.binomial(size=W.size(1), n=1, p=1 - self.p)
        mask = torch.tensor(mask)
        if self.train:
            Z = mask * (X @ W + b).clamp(0)
        else:
            Z = (X @ W + b).clamp(0)
        return Z

    def softmax(self, X, W, b):
        Z = X @ W + b
        return torch.exp(Z) / torch.sum(torch.exp(Z), dim=1, keepdim=True)

    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float)

        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            X = self.relu(X, weight, bias)
        X = self.softmax(X, self.weights[-1], self.biases[-1])

        return X

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        m = ypred.shape[0]
        loss = -torch.sum(ytrue * torch.log(ypred)) / m
        return loss

    def fit(self, X, y, nsteps=1000, bs=100, plot=False):
        X, y = torch.tensor(X), torch.tensor(y)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.lam)

        I = torch.randperm(X.shape[0])
        n = int(np.floor(.9 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

        Ltrain, Lval, Aval = [], [], []
        for i in range(nsteps):
            optimizer.zero_grad()
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]
            output.backward()
            optimizer.step()

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()
