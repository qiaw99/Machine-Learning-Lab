""" ps2_tests.py

Contains tests of the implementations:
- kmeans
- em_gmm
- plot_gmm_solution
- kmeans_agglo
- agglo_dendro

(c) Felix Brockherde, TU Berlin, 2013-2016
"""
import unittest

import numpy as np
import scipy.linalg as la
import pylab as pl
import copy

import ps2_implementation as imp

class TestSheet2(unittest.TestCase):
    X = np.array([[0., 1., 1., 10., 10.25, 11., 10., 10.25, 11.],
                  [0., 0., 1.,  0.,   0.5,  0.,  5.,   5.5,  5.]]).T
    perfect_r = [1,0,1,2,2,1,2,2,2]
    def test_kmeans(self):
        worked1 = False
        worked2 = False

        for _ in range(10):
            mu, r, _ = imp.kmeans(self.X, k=3)
            if (r[0]==r[1]==r[2]!=r[3] and r[3]==r[4]==r[5]!=r[6] and r[6]==r[7]==r[8]):
                worked1 = True

            # test one cluster center
            if (np.linalg.norm(mu[0] - [10.41666, 0.1666]) < 0.1 or
                    np.linalg.norm(mu[1] - [10.41666, 0.1666]) < 0.1 or
                    np.linalg.norm(mu[2] - [10.41666, 0.1666]) < 0.1):
                worked2 = True
            if worked1 and worked2:
                break
        if not worked1:
            raise AssertionError('test_kmeans cluster assignments are wrong.')
        if not worked2:
            raise AssertionError('test_kmeans did not find the correct cluster center.')

    # def test_em_gmm(self):
    #     worked1 = False
    #     worked2 = False
    #     for _ in range(10):
    #         mpi, mu, sigma, _ = imp.em_gmm(self.X, k=3)

    #         # test one cluster center
    #         if (np.linalg.norm(mu[0] - [10.41666, 0.1666]) < 0.1 or
    #             np.linalg.norm(mu[1] - [10.41666, 0.1666]) < 0.1 or
    #             np.linalg.norm(mu[2] - [10.41666, 0.1666]) < 0.1):
    #             worked1 = True
    #         if ((np.abs(np.linalg.det(sigma[0]) - 0.03703) < 0.001 or
    #                np.abs(np.linalg.det(sigma[1]) - 0.03703) < 0.001 or
    #                np.abs(np.linalg.det(sigma[2]) - 0.03703) < 0.001) and
    #                (np.abs(np.linalg.det(sigma[0]) - 0.00925) < 0.001 or
    #                np.abs(np.linalg.det(sigma[1]) - 0.00925) < 0.0001 or
    #                np.abs(np.linalg.det(sigma[2]) - 0.00925) < 0.0001)):
    #             worked2 = True
    #         if worked1 and worked2:
    #             imp.plot_gmm_solution(self.X, mu, sigma)
    #             break

    #     if not worked1:
    #         raise AssertionError('test_em_gmm did not find the correct cluster center.')
    #     if not worked2:
    #         raise AssertionError('test_em_gmm did not find the correct cluster center.')

    def test_agglo(self):
        worked = False
        for _ in range(10):
            mu, r, _ = imp.kmeans(self.X, k=3)
            r = r.flatten()
            r_ = copy.deepcopy(r)
            R, kmloss, mergeidx = imp.kmeans_agglo(self.X, r_)
            mergeidx = np.array(mergeidx, dtype=int)
            if set([int(r[3]), int(r[6])]) == set(mergeidx[0, :]):
                worked = True
                imp.agglo_dendro(kmloss, mergeidx)
                break
        if not worked:
            raise AssertionError('test_agglo: the first merge is not correct.')

if __name__ == '__main__':
    unittest.main()
