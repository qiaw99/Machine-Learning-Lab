""" sheet4_tests.py

(c) Felix Brockherde, TU Berlin, 2013-2016
"""
import unittest

import numpy as np
import torch as tr
import ps4_implementation as imp

class TestSheet4(unittest.TestCase):
    def test_svm_qp(self):
        C = imp.svm_qp(kernel='linear', C=1.)
        np.random.seed(1)
        X_tr = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis])).T
        Y_tr = np.array([1] * 30 + [-1] * 30)
        X_te = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis])).T
        Y_te = np.array([1] * 30 + [-1] * 30)
        C.fit(X_tr, Y_tr)
        Y_pred = C.predict(X_te)
        loss = float(np.sum(np.sign(Y_te) != np.sign(Y_pred)))/float(len(Y_te))
        imp.plot_boundary_2d(X_tr, Y_tr, C)
        print('test case loss', loss)
        self.assertTrue(loss < 0.25, msg='svm_qp: Error. The loss is %.2f and should be below 0.25' % loss)

        
    def test_plot_boundary_2d(self):
        C = imp.svm_sklearn(kernel='gaussian', C=1.)
        np.random.seed(1)
        X_tr = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis])).T
        Y_tr = np.array([1] * 30 + [-1] * 30)
        C.fit(X_tr, Y_tr)
        imp.plot_boundary_2d(X_tr, Y_tr, C)


    def test_neural_network(self):
        X = tr.tensor([[1, 1], [0, 0]], dtype=tr.float)
        y = tr.tensor([[0, 1], [1, 0]], dtype=tr.int)
        W = tr.tensor([[1, .2], [.5, 1]], dtype=tr.float)
        b = tr.tensor([-1, -1], dtype=tr.float)
        m = imp.neural_network(layers=[2,2,2], p=0, lam=0, lr=.1)

        m.fit(X, y, nsteps=1, bs=1, plot=False)

        relu_out = tr.tensor([[.5,.2],[0, 0]])
        self.assertTrue(np.allclose(m.relu(X, W, b), relu_out), msg='neural_network: Error. ReLU output not correct')

        softmax_out = np.array([[0.57444252, 0.42555748], [.5, .5]])
        self.assertTrue(np.allclose(m.softmax(relu_out, W, b), softmax_out), msg='neural_network: Error. Softmax output not correct')

        m.weights = tr.nn.ParameterList([tr.nn.Parameter(W), tr.nn.Parameter(W)])
        m.biases = tr.nn.ParameterList([tr.nn.Parameter(b), tr.nn.Parameter(b)])
        loss_out = 0.7737512125142362
        out = m.forward(X)
        loss = m.loss(out, y).item()
        self.assertTrue(np.isclose(loss_out, loss), msg='neural_network: Error. Loss output not correct')
        self.assertTrue(np.allclose(out.detach().numpy(), softmax_out), msg='neural_network: Error. Network output not correct')

    def test_nn_fit(self):
        np.random.seed(1)
        X_tr = np.hstack((np.random.normal(size=[2, 60]), np.random.normal(size=[2, 60]) + np.array([2., 2.])[:, np.newaxis])).T
        Y_tr = np.array([[1, 0]] * 60 + [[0, 1]] * 60)
        X_te = np.hstack((np.random.normal(size=[2, 60]), np.random.normal(size=[2, 60]) + np.array([2., 2.])[:, np.newaxis])).T
        Y_te = np.array([[1, 0]] * 60 + [[0, 1]] * 60)

        m = imp.neural_network(layers=[2, 100, 2], lr=.1, p=.04, lam=.0)
        m.fit(X_tr, Y_tr, nsteps=1000, bs=20, plot=True)

        Y_pred = m.predict(X_te)
        loss = (Y_pred.argmax(-1) != Y_te.argmax(-1)).mean()
        imp.plot_boundary_2d(X_tr, (-Y_tr).argmax(1), m)
        print('test case loss', loss)
        self.assertTrue(loss < 0.25, msg='neural_network: Error. The loss is %.2f and should be below 0.25' % loss)
        
if __name__ == '__main__':
    unittest.main()
