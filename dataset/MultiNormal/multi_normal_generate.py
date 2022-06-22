from numpy.random import multivariate_normal
import numpy as np
import pickle
import os
from torch.utils.data import Dataset

def cov_function(length_whole, alpha=1):
    # used to create corvariance matrix control by 2 parameters
    cor = np.ones([length_whole, length_whole])
    for i in range(length_whole):
        for j in range(length_whole):
            cor[i, j] = cor[i, j] * np.exp(- np.abs(i - j) / alpha)
    return cor

def anal_solution(X, mu):
    # the input X shape: [batch_size, dimensions]
    # X: np.ndarray
    dims = X.shape[1]
    cor_base = cov_function(dims)
    # Analytical solution can be found and it is convex optimzation if the alpha is known in the exponential formula
    # 1/n \sum_{i=1}^{n} (X-mu) \Sigma (X-mu)^T
    D_mean = np.mean(np.matmul(np.matmul(X - mu, np.linalg.inv(cor_base)), (X - mu).T))
    a = np.linalg.det(cor_base)
    sigma_ested = (np.sqrt(4*a*D_mean + 1) - 1)/(2*a)
    return sigma_ested

def _simu_transform_Gaussian(simu_dim, size, transform, truncate, mode, channels=1):

    data_path = './MultiNormalDataset/' + mode + \
                '/data' + ('_transform' if transform else '') + \
                ('_truncate' if truncate else '') + ('_Dim' + str(simu_dim)) + \
                ('_Chan' + str(channels)) +'.pkl'

    if not os.path.exists(data_path):
        length_whole = simu_dim * channels
        if mode == 'train':
            # mean = np.random.uniform(-2, 2, size=length_whole)
            # # Random semi-definite matrix. Let covariance matrix to be positive-semidefinite
            # cov = np.random.uniform(-1, 1, size=length_whole ** 2).reshape(length_whole, length_whole)
            # cov = np.dot(cov, cov.T)
            # cov = cov + cov.T
            # var = np.diag(1 / np.sqrt(np.diag(cov)))
            # cor = np.matmul(var, cov)
            # cor = np.matmul(cor, var)

            #
            sigma = 1.0
            alpha = 1.0
            mean = np.ones(length_whole) * 1.5
            # cov(si, sj) = \sigma^2 * exp(-||s1 - s2|| / \alpha)
            cor = sigma**2 * cov_function(length_whole=length_whole, alpha=alpha)
        else:
            _data_path = data_path.replace('test', 'train')
            with open(_data_path, 'rb') as f:
                data_GRF_train = pickle.load(f)
                mean, cor = data_GRF_train['mean'], data_GRF_train['cor']
        x = multivariate_normal(mean=mean, cov=cor, size=size)

        if transform:
            # non-linear transformation
            x = np.exp(x) + 1
        if truncate:
            # truncation (0, +inf)
            c = np.max(np.abs(x)) / 1.2
            x[x >= c] = c
        # reshape the dataset
        x = x.reshape(-1, channels, simu_dim)

        data_GRF = {'x': x, 'mean': mean, 'cor': cor}

        with open(data_path, 'wb') as f:
            pickle.dump(data_GRF, f)
        print('Simulation for ' + mode + ' dataset finished!' + ' Path: ' + data_path + ' Shape:', x.shape)

        return x, mean, cor
    else:
        with open(data_path, 'rb') as f:
            data_GRF = pickle.load(f)
        x, mean, cor = data_GRF['x'], data_GRF['mean'], data_GRF['cor']
        if x.shape[0] > size:
            no_shuffle = np.random.randint(0, x.shape[0], size)
            x = x[no_shuffle]
        print('Dataset exists: ' + data_path)
        print(mode + ' Shape: ', x.shape)
        return x, mean, cor



class MultiNormaldataset(Dataset):
    def __init__(self, size, mode, channels=None, simu_dim=None, transform=False, truncate=False):
        assert mode == 'train' or mode == 'test', 'Please input the right mode: train or test.'
        if not os.path.exists('./MultiNormalDataset/' + mode):
            os.makedirs('MultiNormalDataset/' + mode)
        self.x, self.mean, self.cor = _simu_transform_Gaussian(simu_dim, size, transform, truncate, mode, channels)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item]



