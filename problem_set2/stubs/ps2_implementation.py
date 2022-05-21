""" ps2_implementation.py

PUT YOUR NAME HERE:
Qianli Wang
Fengzhou


Write the functions
- kmeans
- kmeans_agglo
- agglo_dendro
- norm_pdf
- em_gmm
- plot_gmm_solution

(c) Felix Brockherde, TU Berlin, 2013
    Translated to Python from Paul Buenau's Matlab scripts
"""

from __future__ import division
from cv2 import detail_BestOf2NearestRangeMatcher  # always use float division
import numpy as np
import scipy
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram
from matplotlib.patches import Ellipse

colors = ['DarkSalmon', 'green', 'yellow', 'dimgray', 'cyan', 'blue', 'magenta']


def kmeans(X, k, max_iter=100):
    """ Performs k-means clustering

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations

    Output:
    mu: (k, n) matrix with each cluster center in one column
    r: assignment vector (k,1)
    """
    n, d = X.shape
    index = np.random.choice(np.arange(n), size=k, replace=False)
    mu = X[index]
    loss = 0
    r = np.ones(n)

    for _ in range(max_iter):
        # Get assigned cluster
        for i in range(n):
            distance = ((X[i, :] - mu)**2).sum(axis=1)**0.5
            index = np.argsort(np.copy(distance))[0]
            r[i] = index

        # Calculate the center of each cluster
        for i in range(k):
            indices = np.where(r==i)
            mu[i, :] = np.mean(X[indices], axis=0)
    for i in range(n):
        loss += (((X[i, :]-mu[int(r[i]), :])**2).sum())**0.5 
    
    return mu, r, loss


def kmeans_agglo(X, r):
    """ Performs agglomerative clustering with k-means criterion

    Input:
    X: (d x n) data matrix with each datapoint in one column
    r: assignment vector

    Output:
    R: (k-1) x n matrix that contains cluster memberships before each step
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    def kmeans_crit(X, r):
        """ Computes k-means criterion

        Input: 
        X: (d x n) data matrix with each datapoint in one column
        r: assignment vector

        Output:
        value: scalar for sum of euclidean distances to cluster centers
        """

        X = X.T
        n, d = X.shape
        loss = 0

        for i in range(n):
            for label in np.unique(r):
                idx = np.argwhere(r==label).flatten()
                tmp = X[i, idx]
                mu = np.mean(tmp, axis=0)
                loss += ((tmp - mu)**2).sum(axis=0)
        return loss

    uniques = np.unique(r)
    _max = uniques[-1]
    k = uniques.shape[0]
    n = X.shape[0]

    if k < 2:
        return

    # Initialization
    R = np.zeros((k-1,n))
    kmloss = np.zeros(k)
    mergeidx = np.zeros((k-1,2))

    # First loss
    kmloss[0] = kmeans_crit(X, r)

    for i in range(k-1):

        # Current cluster
        uniques = np.unique(r)

        # Find a lower-cost merge
        minCost = np.infty
        c1 = c2 = -1
        
        for counter, label in enumerate(uniques):
            for j in range(counter+1, uniques.shape[0]):
                temp = [label if x == uniques[j] else x for x in r]

                if kmeans_crit(X, temp) < minCost:
                    c1, c2 = counter, j
                    minCost = kmeans_crit(X, temp)

        mergeidx[i] = [uniques[c1], uniques[c2]]
        R[i] = r
        r = [_max+1 if x == uniques[c1] or x == uniques[c2] else x for x in r]
        kmloss[i+1] = kmeans_crit(X, r)
        _max += 1

    return R, kmloss, mergeidx


def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering

    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """
    from scipy.cluster import hierarchy
    Z = hierarchy.linkage(np.reshape(kmloss, (len(kmloss), 1)), 'single')
    plt.figure()
    dn = hierarchy.dendrogram(Z)
    plt.xlabel("Cluster index")
    plt.ylabel("kmloss")
    plt.title("Dendrogram of agglomerative clustering with k-means criterion")
    plt.show()

def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: vector for center
    C: covariance matrix

    Output:
    pdf value for each data point
    """
    d = X.shape[0]
    X_mu = X - mu[:, np.newaxis]
    C = C + (0.1 * np.eye(d))
    det_C = np.linalg.det(C)

    tmp1 = np.power((2 * np.pi), -d/2.) * np.power(det_C, -1/2.)
    tmp2 = np.exp(-1/2. * (np.diag(np.dot(X_mu.T, scipy.linalg.solve(C, X_mu, sym_pos=True)))))

    return tmp1 * tmp2

def randomInitCentroids(X, k):
    """ Get random data points as cluster centers
    Input:
    X: (d x n) data matrix with each data point in one column
    k: number of clusters
    Output:
    mu: (d x k) matrix with random cluster center in each column
    """

    d, n = X.shape
    clusters = np.random.choice(n, k, replace=True)
    mu = X[:, clusters]

    return mu

def em_gmm(X, k, max_iter=100, init_kmeans=False, eps=1e-3):
    """ Implements EM for Gaussian Mixture Models

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    init_kmeans: whether kmeans should be used for initialisation
    eps: when log likelihood difference is smaller than eps, terminate loop

    Output:
    pi: 1 x k matrix of priors
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """
    d, n = X.shape
    sigma = []
    if init_kmeans:
        mu, r = kmeans(X,k)
        pi = np.ones(k)
        for idx, cl in enumerate(np.unique(r)):
            pi[idx] = np.sum(r == cl) / n
            sigma.append(np.cov(X[:, np.nonzero(r == cl)[0]]))

    else:
        mu = randomInitCentroids(X, k)
        pi = np.ones(k) / k
        for i in range(k):
            sigma.append(np.eye(d))

    prev_likelihood = 0
    counter = 1
    converged = False
    while not converged:
        ''' The E-Step '''
        gamma = np.zeros([k, n])
        for i in range(k):
            print(X.shape)
            gamma[i, :] = pi[i] * norm_pdf(X, mu[:, i], sigma[i])

        likelihood = np.sum(np.log(np.sum(gamma, axis=0)))
        gamma = gamma / np.sum(gamma, axis=0)  # Normalize gamma

        ''' The M-Step '''
        N = np.sum(gamma, axis=1)
        pi = N / n
        mu = np.dot(X, gamma.T) / N[np.newaxis, :]

        for i in range(k):
            X_zero_mean = X - mu[:, i][:, np.newaxis]
            C = np.dot((gamma[i, :][np.newaxis, :] * X_zero_mean), X_zero_mean.T) / N[i]
            sigma[i] = C

        print('\nIteration:'  + str(counter) + '/' + str(max_iter))
        print('Likelihood: ', likelihood)

        counter = counter + 1
        converged = (np.abs(prev_likelihood - likelihood) < eps) or (counter > max_iter)
        prev_likelihood = likelihood

    return pi, mu, sigma, likelihood

def plot_gmm_solution(X, mu, sigma):
    """ Plots covariance ellipses for GMM

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[0, :], X[1, :])
    ax.scatter(mu[0, :], mu[1, :], marker='x', color='red', s=40)
    ax.set_title('GMM with ' + str(len(sigma)) + ' clusters')
    for k in range(len(sigma)):
        # U, s , Vh = np.linalg.svd(sigma[k])
        eigVal, eigVec = np.linalg.eig(sigma[k])

        orient = np.arctan2(eigVec[1, 0], eigVec[0, 0]) * (180 / np.pi)
        el = Ellipse(xy=mu[:, k], width=2.0 * np.sqrt(eigVal[0]), \
                     height=2.0 * np.sqrt(eigVal[1]), angle=orient, \
                     facecolor='none', edgecolor='red')
        ax.add_patch(el)