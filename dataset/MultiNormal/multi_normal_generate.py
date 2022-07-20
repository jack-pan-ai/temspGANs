'''
Used to generation sample from Matern covariance function
'''

from numpy.random import multivariate_normal
import numpy as np
import pickle
import os
from torch.utils.data import Dataset
from scipy import special

# def anal_solution(X, mu, channels=None):
#     # the input X shape: [batch_size, dimensions]
#     # X: np.ndarray
#     dims = X.shape[1]
#     cov_base = cov_function(dims, channels=channels)
#     cov_base_inv = np.linalg.inv(cov_base)
#     # Analytical solution can be found and it is convex optimzation if the alpha is known in the exponential formula
#     X_decentralized = X - mu
#     sum_sigma = 0
#     for i in range(X.shape[0]):
#         sum_sigma += X_decentralized[i:(i+1), :] @ cov_base_inv @ X_decentralized[i:(i+1), :].T
#     sigma_ested = sum_sigma/(X.shape[0] * X.shape[1])
#     return sigma_ested

def matern_cov(distance, nu, rho):
    item1 = 2**(1 - nu)/special.gamma(nu)
    item2 = (np.sqrt(2*nu) * distance/rho)**nu
    item3 = special.kv(nu, np.sqrt(2*nu) * distance/rho)
    return item1 * item2 * item3

def cov_function(length_whole, tau, nu, rho,channels=None):
    # used to create covvariance matrix control by 2 parameters
    if channels==1:
        cov = np.ones([length_whole, length_whole])
        for i in range(length_whole):
            for j in range(length_whole):
                distance = np.abs(i-j)
                cov[i, j] = cov[i, j] * matern_cov(distance=distance, nu=nu, rho=rho)
    else:
        cov = np.ones([length_whole, length_whole])
        for i in range(length_whole):
            for j in range(length_whole):
                # (x, y)
                x_i = i // channels
                y_i = i % channels
                x_j = j // channels
                y_j = j % channels
                # L-2 distance
                distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j)**2) / (length_whole / channels)
                if distance == 0:
                    cov[i, j] = tau**2
                else:
                    cov[i, j] = tau**2 * matern_cov(distance=distance, nu=nu, rho=rho)
    return cov

def _simu_transform_Gaussian(simu_dim, size, transform, truncate, mode, channels=1, args=None):

    data_path = './MultiNormalDataset/' + mode + \
                '/data' + ('_transform' if transform else '') + \
                ('_truncate' if truncate else '') + ('_Dim' + str(simu_dim)) + \
                ('_Chan' + str(channels)) +'.pkl'

    if not os.path.exists(data_path):
        length_whole = simu_dim * channels
        if mode == 'train':
            tau = args.tau
            nu = args.nu
            rho = args.rho
            mean = np.ones(length_whole) * 1.5
            # cov(si, sj) = \sigma^2 * exp(-||s1 - s2|| / \alpha)
            cov = cov_function(length_whole=length_whole, tau=tau, nu=nu, rho=rho, channels=channels)
        else:
            _data_path = data_path.replace('test', 'train')
            with open(_data_path, 'rb') as f:
                data_GRF_train = pickle.load(f)
                mean, cov = data_GRF_train['mean'], data_GRF_train['cov']
        x = multivariate_normal(mean=mean, cov=cov, size=size) # [batch_size, whole_length]

        if transform:
            # non-linear transformation
            x = np.exp(x) + 1
        if truncate:
            # truncation (0, +inf)
            c = np.max(np.abs(x)) / 1.2
            x[x >= c] = c
        # reshape the dataset
        x = x.reshape([-1, channels, simu_dim])

        data_GRF = {'x': x, 'mean': mean, 'cov': cov}

        with open(data_path, 'wb') as f:
            pickle.dump(data_GRF, f)
        print('Simulation for ' + mode + ' dataset finished!' + ' Path: ' + data_path + ' Shape:', x.shape)

        return x, mean, cov
    else:
        with open(data_path, 'rb') as f:
            data_GRF = pickle.load(f)
        x, mean, cov = data_GRF['x'], data_GRF['mean'], data_GRF['cov']
        if x.shape[0] > size:
            no_shuffle = np.random.randint(0, x.shape[0], size)
            x = x[no_shuffle]
        print('Dataset exists: ' + data_path)
        print(mode + ' Shape: ', x.shape)
        return x, mean, cov



class MultiNormaldataset(Dataset):
    def __init__(self, size, mode, channels=None, simu_dim=None, transform=False, truncate=False, args=None):
        assert mode == 'train' or mode == 'test', 'Please input the right mode: train or test.'
        if not os.path.exists('./MultiNormalDataset/' + mode):
            os.makedirs('MultiNormalDataset/' + mode)
        self.x, self.mean, self.cov = _simu_transform_Gaussian(simu_dim, size, transform, truncate, mode, channels, args)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item]

if __name__ == '__main__':
    a = cov_function(length_whole = 64, tau=1.0, nu=0.5, rho = 1.0, channels=8)
    print(np.isnan(a))
    print(a)
    # b = matern_cov(1.0, 0.5, 1.0)
    # print(b)

