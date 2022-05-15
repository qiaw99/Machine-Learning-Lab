import ps2_implementation as imp
import numpy as np
import copy
from matplotlib import pyplot as plt

colors = ['DarkSalmon', 'green', 'yellow', 'dimgray', 'cyan', 'blue', 'magenta']
# , 'red', 'lightblue', 'lime'

def assignment7():
    # X = (2, 500)
    X = np.load("../data/5_gaussians.npy")
    plt.scatter(X.T[:, 0], X.T[:, 1])
    plt.title("5 gaussians data")
    plt.show()

    all_mu = []
    all_loss = []
    all_r = []

    for i in range(2, 8): 
        mu, r, loss = imp.kmeans(X.T, k=i)
        all_mu.append(mu)
        all_loss.append(loss)
        all_r.append(r)

        r = r.flatten()
        r_ = copy.deepcopy(r)
        R, kmloss, mergeidx = imp.kmeans_agglo(X.T, r_)
        mergeidx = np.array(mergeidx, dtype=int)
        imp.agglo_dendro(kmloss, mergeidx)

    all_r = np.array(all_r)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(3*2, 5*5))
    id_x, id_y = 0, 0
    for i in range(2, 8):
        for idx, cl in enumerate(np.unique(all_r[i-2])):
            X_idx = X[:, np.nonzero(all_r[i-2] == cl)[0]]
            axes[id_x, id_y].scatter(X_idx[0, :], X_idx[1, :], c=colors[idx])
            axes[id_x, id_y].set_title('k-means with '+ str(i) + ' clusters')
        
        id_y += 1
        if id_y == 2:
            id_y = 0
            id_x += 1
    plt.show()

def assignment8():
    pass

def assignment9():
    pass

def assignment10():
    with np.load('../data/lab_data.npz') as data:
        X = data["X"]
        Y = data["Y"]
        print(X.shape, Y.shape)

if __name__ == "__main__":
    assignment10()