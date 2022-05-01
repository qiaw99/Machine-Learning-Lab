""" sheet1_implementation.py

PUT YOUR NAMES HERE:
Qianli Wang 
Feng Zhou


Write the functions
- usps
- outliers
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2021
"""
from turtle import color
import numpy as np
import ps1_implementation as imp
import scipy.io
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def usps():
    ''' performs the usps analysis for assignment 5'''
    mat = scipy.io.loadmat("../data/usps.mat")
    data_labels = mat["data_labels"]
    data_patterns = mat["data_patterns"]
    print(data_labels.shape)
    print(data_patterns.shape)
    pca = imp.PCA(data_patterns)
    eigen_values = pca.D

    
    flag = True
    plt.figure(figsize=(20,25))
    while(flag):
        plt.subplot(4,4,1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # All principal values
        plt.bar(np.arange(0, eigen_values.shape[0]), height=np.asarray(eigen_values, float), color='r')
        plt.ylabel("Eigen values")
        plt.xlim([0,60])
        plt.ylim([0, 250])
        plt.title("All principal values of original image")

        plt.subplot(4,4,2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.bar(np.arange(0, 25), height=np.asarray(eigen_values[:25], float))
        plt.ylabel("Eigen values")
        plt.xlim([0,30])
        plt.ylim([0, 250])
        plt.title("First 25 principal values")

        plt.subplot(4,4,3)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.asarray(data_patterns, float))
        plt.title("original image")

        pca.project(np.copy(data_patterns), 5)
        img = pca.denoise(np.copy(data_patterns), 5)
        plt.subplot(4,4,4)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.asarray(img,float))
        plt.title("Denoised image")

        

        # low gaussian noise
        low_noise = np.random.randn(256, 2007) * 0.01
        low_gaussian_data = np.copy(data_patterns)
        low_gaussian_data += low_noise
        pca1 = imp.PCA(low_gaussian_data)
        eigen_values = pca1.D
        plt.subplot(4,4,5)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # All principal values
        plt.bar(np.arange(0, eigen_values.shape[0]), height=np.asarray(eigen_values, float), color='r')
        plt.ylabel("Eigen values")
        plt.xlim([0,60])
        plt.ylim([0, 250])
        plt.title("All principal values of image with low gaussian")

        plt.subplot(4,4,6)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.bar(np.arange(0, 25), height=np.asarray(eigen_values[:25], float))
        plt.ylabel("Eigen values")
        plt.xlim([0,30])
        plt.ylim([0, 250])
        plt.title("First 25 principal values")
        

        plt.subplot(4,4,7)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.asarray(low_gaussian_data, float))
        plt.title("Noisy image")

        pca1.project(np.copy(low_gaussian_data), 5)
        img = pca.denoise(np.copy(low_gaussian_data), 5)
        plt.subplot(4,4,8)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.asarray(img,float))
        plt.title("Denoised image")

        # high gaussian noise
        high_noise = np.random.randn(256, 2007) * 0.5
        high_gaussian_data = np.copy(data_patterns)
        high_gaussian_data += high_noise
        pca2 = imp.PCA(high_gaussian_data)
        eigen_values = pca2.D
        plt.subplot(4,4,9)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # All principal values
        plt.bar(np.arange(0, eigen_values.shape[0]), height=np.asarray(eigen_values, float), color='r')
        plt.ylabel("Eigen values")
        plt.xlim([0,60])
        plt.ylim([0, 250])
        plt.title("All principal values of image with high gaussian")

        plt.subplot(4,4,10)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.bar(np.arange(0, 25), height=np.asarray(eigen_values[:25], float))
        plt.ylabel("Eigen values")
        plt.xlim([0,30])
        plt.ylim([0, 250])
        plt.title("First 25 principal values")
        

        plt.subplot(4,4,11)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.asarray(high_gaussian_data, float))
        plt.title("Noisy image")

        pca2.project(np.copy(high_gaussian_data), 5)
        img = pca.denoise(np.copy(high_gaussian_data), 5)
        plt.subplot(4,4,12)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.asarray(img,float))
        plt.title("Denoised image")

    	# outliers
        outlier_noise = np.random.randn(256, 2007) 
        outlier_data = np.copy(data_patterns)
        outlier_data += outlier_noise
        pca3 = imp.PCA(outlier_data)
        eigen_values = pca3.D
        plt.subplot(4,4,13)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # All principal values
        plt.bar(np.arange(0, eigen_values.shape[0]), height=np.asarray(eigen_values, float), color='r')
        plt.ylabel("Eigen values")
        plt.xlim([0,60])
        plt.ylim([0, 250])
        plt.title("All principal values of image with outlier")

        plt.subplot(4,4,14)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.bar(np.arange(0, 25), height=np.asarray(eigen_values[:25], float))
        plt.ylabel("Eigen values")
        plt.xlim([0,30])
        plt.ylim([0, 250])
        plt.title("First 25 principal values")
        

        plt.subplot(4,4,15)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.asarray(outlier_data, float))
        plt.title("Noisy image")

        pca3.project(np.copy(outlier_data), 5)
        img = pca.denoise(np.copy(outlier_data), 5)
        plt.subplot(4,4,16)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.asarray(img,float))
        plt.title("Denoised image")

        plt.show()

        flag = False


    


def outliers_calc():
    ''' outlier analysis for assignment 6'''
    # np.savez_compressed('outliers.npz', var1=var1, var2=var2, ...)


def outliers_disp():
    ''' display the boxplots'''
    # results = np.load('outliers.npz')


def lle_visualize(dataset='flatroll'):
    ''' visualization of LLE for assignment 7'''




def lle_noise():
    ''' LLE under noise for assignment 8'''    
    data = np.load("../data/flatroll_data.npz")
    X_flat = data['Xflat']
    true_embedding = data['true_embedding']
    print(X_flat.shape, true_embedding.shape)

    # get 2 noisy datasets
    gaussian_1 = np.random.normal(0, 0.2, size=(2, 1000))
    gaussian_2 = np.random.normal(0, 1.8, size=(2, 1000))
    data1 = np.copy(X_flat) + gaussian_1
    data2 = np.copy(X_flat) + gaussian_2

    # plotting for var=0.2
    # f = plt.figure(figsize=(14,8))
    # ax = f.add_subplot(1,2,1)

    # embedding = imp.lle(data1.T, 1, 'knn', k=15)
    # # bad k(too large)
    # # embedding = imp.lle(data1.T, 1, 'knn', k=100)

    # ax.scatter(data1.T[:, 0], data1.T[:, 1], c=true_embedding)
    # ax.set_title('Dataset1')
    # ax.set_xticks([], [])
    # ax.set_yticks([], [])

    # ax = f.add_subplot(1,2,2)
    # ax.scatter(embedding[:,-1],np.zeros(shape=embedding[:,-1].shape), c=true_embedding)
    # ax.set_title('Dataset1 embedding with k = 5')
    # ax.set_xticks([], [])
    # ax.set_yticks([], [])
    # plt.show()

    f = plt.figure(figsize=(20,3))
    for t,k in enumerate([i for i in range(1, 50, 5)]):
        embedding = imp.lle(data2.T, 1, 'knn', k=k)
        ax = f.add_subplot(1,11,t+1)
        ax.set_title('k=%d'%(k))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(embedding[:,0],np.zeros(shape=embedding[:,0].shape),c=true_embedding.T)    
    plt.show()




if __name__ == "__main__":
    lle_noise()