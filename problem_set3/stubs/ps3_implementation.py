""" ps3_implementation.py

PUT YOUR NAME HERE:
Qianli Wang, Feng Zhou


Write the functions
- cv
- zero_one_loss
- krr
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

def zero_one_loss(y_true, y_pred):
    return np.count_nonzero(y_true != y_pred) / float(y_true.shape[0])

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5, roc_f=False, verbose = True):
    
    def updateProgress(total, progress, details, remain_time):
        length, msg = 100, ""
        progress = float(progress) / float(total)
        if progress >= 1.:
            progress, msg = 1, "\r\n"
        current = int(round(length * progress))
        print(details)
        print("\r[{}] {:.0f}% {}, remain:{}".format(">" * current + "-" * (length - current), round(progress * 100, 0), msg, remain_time))
        print()

    last_times = []
    def get_remaining_time(step, total, time):
        last_times.append(time)
        len_last_t = len(last_times)
        if len_last_t > 120:
            last_times.pop(0)
        mean_time = np.mean(last_times)
        remain_s_tot = mean_time * (total - step + 1)
        minutes = remain_s_tot // 60
        seconds = remain_s_tot % 60
        return "{}m {}s".format(minutes, seconds)

    n, d = X.shape
    indexes = np.arange(0, n)
    test_size = int(n/nfolds)

    combinations_params = list(it.product(*(params[Name] for Name in list(params))))
    n_combinations = len(combinations_params)
    error_combinations = np.zeros(n_combinations)

    runs = n_combinations * nrepetitions
    progress = 0
    for cur_index, local_parameter in enumerate(combinations_params):
        current_time = time.time()
        e_r_error = 0
        method_fold = method(*local_parameter)
        for i in range(nrepetitions):
            indexes_shuffled = indexes.copy()
            np.random.shuffle(indexes_shuffled)
            e_delta = 0
            for cur_fold in range(nfolds):
                if cur_fold == (nfolds - 1):
                    test_fold_indexes = indexes_shuffled[(cur_fold * test_size):]
                    train_fold_indexes = np.delete(indexes_shuffled,
                                                   np.nonzero(np.isin(indexes_shuffled, test_fold_indexes)))
                else:
                    test_fold_indexes = indexes_shuffled[(cur_fold * test_size):(
                                cur_fold * test_size + test_size)]
                    train_fold_indexes = np.delete(indexes_shuffled,
                                                   np.nonzero(np.isin(indexes_shuffled, test_fold_indexes)))

                Y_test = y[test_fold_indexes]
                X_test = X[test_fold_indexes, :]

                X_train = X[train_fold_indexes, :]
                Y_train = y[train_fold_indexes]

                method_fold.fit(X_train, Y_train)
                y_pred = method_fold.predict(X_test)

                e_delta += loss_function(Y_test, y_pred)

            e_r_error += (e_delta/nfolds)
            last_t = time.time() - current_time

            if verbose:
                details = "kernel:{},kernelparam:{},regularizer:{}".format(local_parameter[0], local_parameter[1], local_parameter[2])
                updateProgress(runs, progress + 1, details, get_remaining_time(progress + 1, runs, last_t))
                progress += 1

        final_error = e_r_error / nrepetitions
        if roc_f==False:
            error_combinations[cur_index] = final_error

    if roc_f==False:
        best_param_combination = combinations_params[np.argmin(error_combinations)]
        best_method = method(*best_param_combination)
        best_method.cvloss = np.min(error_combinations)
        best_method.fit(X, y)

        print('best method:', best_param_combination)
        print('best method loss: {}'.format(best_method.cvloss))
    else:
        best_method = method(*combinations_params) #combinations_params are already the best
        best_method.tp = final_error[0]
        best_method.fp = final_error[1]

    return best_method
  
class krr():
    ''' your header here!
    '''
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization
        self.trainX = None
        self.alpha = None 

    def getKernel(self, X):
        if self.kernel == 'linear':
            return X.T * X
        elif self.kernel == 'polynomial':
            return (X.T * X + 1) ** self.kernelparameter
        elif self.kernel == 'gaussian':
            X = X.reshape(-1, 1)
            ret = np.exp(-1 * (X.T - X) ** 2/ 2 * self.kernelparameter ** 2)
            return ret

    def getKernelPrediction(self, X):
        if self.kernel == 'linear':
            return self.trainX.T * X
        elif self.kernel == 'polynomial':
            return (self.trainX.T * X + 1) ** self.kernelparameter
        elif self.kernel == 'gaussian':
            return np.exp(-1 * (self.trainX.T - X) ** 2 / 2 * self.kernelparameter ** 2)

    def loocv(self, K, Y):
        eigval, U = la.eig(K)
        U = U.real
        L = np.diag(eigval).real
        UL = np.dot(U, L)

        C3 = np.logspace(-5, 5, 100)
        I3 = np.eye(len(L))
        CI3 = np.einsum('ij,k->kij', I3, C3)

        CI3L = CI3 + L
        dt = np.dtype(np.float32)
        apinv = list(map(lambda n: la.pinv(n), CI3L))
        apinv = np.asarray(apinv, dtype=dt)

        ULCI3 = np.einsum('lj,ijk->ilk', UL, apinv)
        ULCI3UT = np.einsum('ikj,jl->ikl', ULCI3, U.T)
        S = ULCI3UT
        Sdiag = np.einsum('kii->ki', S)
        err = np.mean(np.square((Y - np.dot(S, Y)) * ((1 - Sdiag) ** -1)), axis=1)

        return C3[np.where(err == min(err))]

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        ''' your header here!
        '''
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization
        self.trainX = X

        K = self.getKernel(X)
        if self.regularization == 0:
            self.regularization = self.loocv(K, y)
        mat = K + self.regularization * np.eye(len(X))
        self.alpha = np.linalg.inv(mat) @ y


    def predict(self, X):
        ''' your header here!
        '''
        return np.sum(self.alpha * self.getKernelPrediction(X), axis=1)
