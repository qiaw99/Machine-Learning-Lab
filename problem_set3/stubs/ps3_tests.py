""" ps3_tests.py

Contains tests of the implementations:
- pca
- lle
- gammaidx

(c) Daniel Bartz, TU Berlin, 2013
(c) Felix Brockherde, TU Berlin, 2016
"""
import unittest

import numpy as np
import scipy.linalg as la
import pylab as pl

import ps3_implementation as imp

def squared_error_loss(y_true, y_pred):
    ''' returns the squared error loss
    '''
    assert(len(y_true) == len(y_pred))
    loss = np.mean( (y_true - y_pred)**2 )
    return loss

def noisysincfunction(N, noise):
    ''' noisysincfunction - generate data from the "noisy sinc function"
        % usage
        %     [X, Y] = noisysincfunction(N, noise)
        %
        % input
        %     N: number of data points
        %     noise: standard variation of the noise
        %
        % output
        %     X: (1, N)-matrix uniformly sampled in -2pi, pi
        %     Y: (1, N)-matrix equal to sinc(X) + noise
        %
        % description
        %     Generates N points from the noisy sinc function
        %
        %        X ~ uniformly in [-2pi, pi]
        %        Y = sinc(X) + eps, eps ~ Normal(0, noise.^2)
        %
        % author
        %     Mikio Braun
    '''
    X = np.sort(2 * np.pi * np.random.rand(1, N) ) - np.pi
    Y = np.sinc(X) + noise * np.random.randn(1, N)
    return X.reshape(-1, 1), Y.flatten()

class TestSheet3(unittest.TestCase):
    def test_krr(self):
        '''
            tests the class krr
        '''
        Xtr, Ytr = noisysincfunction(100, 0.1)
        Xte = np.arange( -np.pi, np.pi, 0.01 ).reshape(-1, 1)

        pl.figure()
        kernels = ['gaussian','polynomial','linear']
        titles = ['gaussian','polynomial','linear']
        params = [0.5,6,0]
        regularizations = [ 0.01,0.01,0.01]
        for i in range(3):
            for j in range(2):
                pl.subplot(2,3,1+i+3*j)
                if j==0:
                    krr = imp.krr(kernel=kernels[i],
                            kernelparameter=params[i],
                            regularization=regularizations[i])
                    krr.fit(Xtr,Ytr)
                if j==1:
                    krr = imp.krr(kernel=kernels[i],
                            kernelparameter=params[i],
                            regularization=0)
                    krr.fit(Xtr,Ytr)
                ypred = krr.predict(Xte)
                pl.plot(Xtr,Ytr)
                pl.plot(Xte,ypred)
                if j==0 and i == 0:
                    pl.ylabel('fixed regularization')
                if j==1 and i == 0:
                    pl.ylabel('reg. by efficent cv')
                pl.title( titles[i] )
        pl.show()

    def test_cv(self):
        Xtr, Ytr = noisysincfunction(100, 0.1)
        Xte = np.arange( -np.pi, np.pi, 0.01 ).reshape(-1, 1)

        pl.figure()
        pl.subplot(1,2,1)
        params = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-4,4,20),
                      'regularization': np.logspace(-2,2,10) }
        cvkrr = imp.cv(Xtr, Ytr, imp.krr, params, loss_function=squared_error_loss,
                        nrepetitions=2)
        ypred = cvkrr.predict(Xte)
        print('Regularization range: 10**-4 .. 10**4')
        print('Gaussian kernel parameter: ', cvkrr.kernelparameter)
        print('Regularization paramter: ', cvkrr.regularization)

        pl.plot(Xtr,Ytr)
        pl.plot(Xte,ypred)

        pl.subplot(1,2,2)
        params = { 'kernel': ['gaussian'], 'kernelparameter': np.logspace(-2,2,10),
                      'regularization': [0]}
        cvkrr = imp.cv(Xtr, Ytr, imp.krr, params, loss_function=squared_error_loss,
                        nrepetitions=2)
        ypred = cvkrr.predict(Xte)
        print('Regularization via efficient leave on out')
        print('Kernel parameter: ', cvkrr.kernelparameter)
        print('Regularization paramter: ', cvkrr.regularization)

        pl.plot(Xtr,Ytr)
        pl.plot(Xte,ypred)
        pl.show()

if __name__ == '__main__':
    unittest.main()
