""" sheet1_tests.py

Contains tests of the implementations:
- pca
- lle
- gammaidx

Written by
Felix Brockherde, TU Berlin, 2013-2016
"""
import unittest

import numpy as np
import numpy.testing as npt
from scipy.linalg import expm
import matplotlib.pyplot as plt

import ps1_implementation as imp

class TestSheet1(unittest.TestCase):

    def test_pca(self):
        X = np.array([[ -2.133268233289599,   0.903819474847349,   2.217823388231679, -0.444779660856219,
                        -0.661480010318842,  -0.163814281248453,  -0.608167714051449,  0.949391996219125],
                      [ -1.273486742804804,  -1.270450725314960,  -2.873297536940942,   1.819616794091556,
                        -2.617784834189455,   1.706200163080549,   0.196983250752276,   0.501491995499840],
                      [ -0.935406638147949,   0.298594472836292,   1.520579082270122,  -1.390457671168661,
                        -1.180253547776717,  -0.194988736923602,  -0.645052874385757,  -1.400566775105519]]).T
        m = 2;
        correct_Z = np.array([  [   -0.264248351888547, 1.29695602132309, 3.59711235194654, -2.45930603721054,
                                    1.33335186376208, -1.82020953874395, -0.85747383354342, -0.82618247564525],
                                [   2.25344911712941, -0.601279409451719, -1.28967825406348, -0.45229125158068,
                                    1.82830152899142, -1.04090644990666, 0.213476150303194, -0.911071431421484]]).T

        correct_U = np.array([  [   0.365560300980795,  -0.796515907521792,  -0.481589114714573],
                                [   -0.855143149302632,  -0.491716059542403,   0.164150878733159],
                                [  0.367553887950606,  -0.351820587590931,   0.860886992351241]] )

        correct_D = np.array(   [ 3.892593483673686,   1.801314737893267,   0.356275626798182 ])

        correct_X_denoised = np.array([[-1.88406616, -1.35842791, -1.38087939],
                                         [ 0.96048487, -1.28976527,  0.19729962],
                                         [ 2.34965134, -2.91823143,  1.28492391],
                                         [-0.53132686,  1.84911663, -1.23574621],
                                         [-0.96141012, -2.51555289, -0.64409954],
                                         [ 0.17114282,  1.59202918, -0.79375686],
                                         [-0.47605492,  0.15195227, -0.88121723],
                                         [ 0.43110399,  0.67815178, -0.47407698]])

        pca = imp.PCA(X)
        U, D = pca.U, pca.D
        Z = pca.project(X, m=2)
        X_denoised = pca.denoise(X, m=2)
        npt.assert_equal(Z.shape, correct_Z.shape, err_msg='Matrix Z does not have the correct shape')
        npt.assert_equal(U.shape, correct_U.shape, err_msg='Matrix U does not have the correct shape')
        npt.assert_equal(D.shape, correct_D.shape, err_msg='Matrix D does not have the correct shape')
        npt.assert_equal(X_denoised.shape, correct_X_denoised.shape, err_msg='Denoised matrix does not have the correct shape')

        if not (np.allclose(U[:,0], correct_U[:,0]) or np.allclose(U[:,0], -correct_U[:,0])):
            raise AssertionError('First principal component is not correct.')
        if not (np.allclose(U[:,1], correct_U[:,1]) or np.allclose(U[:,1], -correct_U[:,1])):
            raise AssertionError('Second principal component is not correct.')
        if not (np.allclose(U[:,2], correct_U[:,2]) or np.allclose(U[:,2], -correct_U[:,2])):
            raise AssertionError('Third principal component is not correct.')
        if not (np.allclose(Z[:, 0], correct_Z[:, 0]) or np.allclose(Z[:, 0], -correct_Z[:, 0])):
            raise AssertionError('First projected dimension is not correct.')
        if not (np.allclose(Z[:, 1], correct_Z[:, 1]) or np.allclose(Z[:, 1], -correct_Z[:, 1])):
            raise AssertionError('Second projected dimension is not correct.')
        npt.assert_allclose(D, correct_D, err_msg='Matrix D is not correct')
        npt.assert_allclose(X_denoised, correct_X_denoised, err_msg='Denoised matrix is not correct')
    
    def test_gammaidx(self):
        X = np.array([  [   0.5376671395461, -2.25884686100365, 0.318765239858981, -0.433592022305684, 3.57839693972576,
                            -1.34988694015652, 0.725404224946106, 0.714742903826096, -0.124144348216312, 1.40903448980048,
                            0.67149713360808, 0.717238651328838, 0.488893770311789, 0.726885133383238, 0.293871467096658,
                            0.888395631757642, -1.06887045816803, -2.9442841619949, 0.325190539456198, 1.37029854009523],
                        [   1.83388501459509, 0.862173320368121, -1.30768829630527, 0.34262446653865, 2.76943702988488,
                            3.03492346633185, -0.0630548731896562, -0.204966058299775, 1.48969760778546, 1.41719241342961,
                            -1.20748692268504, 1.63023528916473, 1.03469300991786, -0.303440924786016, -0.787282803758638,
                            -1.14707010696915, -0.809498694424876, 1.4383802928151, -0.754928319169703, -1.7115164188537]]).T

        k = 3;

        correct_gamma = np.array([ 0.606051220224367, 1.61505686776722, 0.480161964450438, 1.18975154873627,
                                    2.93910520141032, 2.15531724762712, 0.393996268071324, 0.30516080506303,
                                    0.787481421847747, 0.895402545799062, 0.385599174039363, 0.544395897115756,
                                    0.73397995201338, 0.314642851266896, 0.376994725474732, 0.501091387197748,
                                    1.3579045507961, 1.96372676400505, 0.389228251829715, 0.910065898315003])

        gamma = imp.gammaidx(X, k)
        npt.assert_equal(gamma.shape, correct_gamma.shape, err_msg='gamma does not have the correct shape')

        npt.assert_allclose(gamma, correct_gamma, err_msg='gamma is not correct')

    def randrot(self, d):
        '''generate random orthogonal matrix'''
        M = 100. * (np.random.rand(d, d) - 0.5)
        M = 0.5 * (M - M.T);
        R = expm(M);
        return R

    def plot(self, Xt, Xp, n_rule):
        plt.figure(figsize=(14, 8))

        plt.subplot(1, 3, 1)
        plt.scatter(Xt[:, 0], Xt[:, 1], 30)
        plt.title('True 2D manifold')
        plt.ylabel(r'$x_2$')
        plt.xticks([], [])
        plt.yticks([], [])

        plt.subplot(1, 3, 2);
        plt.scatter(Xp[:, 0], Xp[:, 1], 30, Xt[:, 0]);
        plt.title(n_rule + r': embedding colored via $x_1$');
        plt.xlabel(r'$x_1$')
        plt.xticks([], [])
        plt.yticks([], [])

        plt.subplot(1, 3, 3);
        plt.scatter(Xp[:, 0], Xp[:, 1], 30, Xt[:, 1]);
        plt.title(n_rule + r': embedding colored via $x_2$');
        plt.xticks([], [])
        plt.yticks([], [])

        plt.show()
    
    def test_lle(self):
        n = 500
        Xt = 10. * np.random.rand(n, 2);
        X = np.append(Xt, 0.5 * np.random.randn(n, 8), 1);

        # Rotate data randomly.
        X = np.dot(X, self.randrot(10).T);

        with self.subTest(n_rule='knn', k=30):
            Xp = imp.lle(X, 2, n_rule='knn', k=30, tol=1e-3)
            # self.plot(Xt, Xp, 'knn')
        with self.subTest(n_rule='eps-ball', epsilon=5.):
            Xp = imp.lle(X, 2, n_rule='eps-ball', epsilon=5., tol=1e-3)
            # self.plot(Xt, Xp, 'eps-ball')
        with self.subTest(n_rule='eps-ball', epsilon=0.5):
            with self.assertRaises(ValueError, msg='Graph should not be connected and raise ValueError.'):
                # imp.lle(X, 2, n_rule='eps-ball', epsilon=0.5, tol=1e-3)
                imp.lle(X, 2, n_rule='eps-ball', tol=1e-3)
                

    def test_auc(self):
        res = imp.auc(np.array([-1, -1, -1, +1, +1]), np.array([0.3, 0.4, 0.5, 0.6, 0.7]))
        npt.assert_allclose(res, 1.0, err_msg='Separable dataset should give AUC of 1.0')
        print('res = %g should be 1.0' % res)
        res = imp.auc(np.array([-1, -1, -1, +1, +1, +1]), np.array([0.3, 0.4, 0.6, 0.5, 0.7, 0.8]), plot=False)
        npt.assert_allclose(res, 0.89, rtol=0.05, atol=5e-2, err_msg='AUC example failed.')
        print('res = %g should be 0.89' % res)
        res = imp.auc(np.array([+1, -1, -1, +1, +1, -1]), np.array([0.3, 0.4, 0.6, 0.5, 0.7, 0.8]), plot=False)
        npt.assert_allclose(res, 1./3., rtol=0.05, atol=5e-2, err_msg='AUC example failed.')
        print('res = %g should be 1/3' % res)

if __name__ == '__main__':
    unittest.main()
