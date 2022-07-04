from numpy.random import multivariate_normal
import numpy as np
import pickle
import os
from torch.utils.data import Dataset

def cov_function(length_whole, alpha=0.1, channels=None):
    # used to create corvariance matrix control by 2 parameters
    if channels==1:
        cor = np.ones([length_whole, length_whole])
        for i in range(length_whole):
            for j in range(length_whole):
                cor[i, j] = cor[i, j] * np.exp(- np.abs(i - j) / alpha)
    else:
        cor = np.ones([length_whole, length_whole])
        for i in range(length_whole):
            for j in range(length_whole):
                # (x, y)
                x_i = i // channels
                y_i = i % channels
                x_j = j // channels
                y_j = j % channels
                # L-2 distance
                distances = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j)**2)
                cor[i, j] = cor[i, j] * np.exp(- distances / alpha)
    return cor

def anal_solution(X, mu, channels=None):
    # the input X shape: [batch_size, dimensions]
    # X: np.ndarray
    dims = X.shape[1]
    cor_base = cov_function(dims, channels=channels)
    cor_base_inv = np.linalg.inv(cor_base)
    # Analytical solution can be found and it is convex optimzation if the alpha is known in the exponential formula
    X_decentralized = X - mu
    sum_sigma = 0
    for i in range(X.shape[0]):
        sum_sigma += X_decentralized[i:(i+1), :] @ cor_base_inv @ X_decentralized[i:(i+1), :].T
    sigma_ested = sum_sigma/(X.shape[0] * X.shape[1])
    return sigma_ested

def _simu_transform_Gaussian(simu_dim, size, transform, truncate, mode, channels=1, args=None):

    data_path = './MultiNormalDataset/' + mode + \
                '/data' + ('_transform' if transform else '') + \
                ('_truncate' if truncate else '') + ('_Dim' + str(simu_dim)) + \
                ('_Chan' + str(channels)) +'.pkl'

    if not os.path.exists(data_path):
        length_whole = simu_dim * channels
        if mode == 'train':
            sigma = args.sigma
            alpha = args.alpha
            mean = np.ones(length_whole) * 1.5
            # cov(si, sj) = \sigma^2 * exp(-||s1 - s2|| / \alpha)
            cor = sigma**2 * cov_function(length_whole=length_whole, alpha=alpha, channels=channels)
        else:
            _data_path = data_path.replace('test', 'train')
            with open(_data_path, 'rb') as f:
                data_GRF_train = pickle.load(f)
                mean, cor = data_GRF_train['mean'], data_GRF_train['cor']
        x = multivariate_normal(mean=mean, cov=cor, size=size) # [batch_size, whole_length]

        if transform:
            # non-linear transformation
            x = np.exp(x) + 1
        if truncate:
            # truncation (0, +inf)
            c = np.max(np.abs(x)) / 1.2
            x[x >= c] = c
        # reshape the dataset
        x = x.reshape([-1, channels, simu_dim])

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
    def __init__(self, size, mode, channels=None, simu_dim=None, transform=False, truncate=False, args=None):
        assert mode == 'train' or mode == 'test', 'Please input the right mode: train or test.'
        if not os.path.exists('./MultiNormalDataset/' + mode):
            os.makedirs('MultiNormalDataset/' + mode)
        self.x, self.mean, self.cor = _simu_transform_Gaussian(simu_dim, size, transform, truncate, mode, channels, args)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item]



