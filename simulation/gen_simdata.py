import numpy as np
import pandas as pd

from scipy.linalg import solve
from scipy.stats import multivariate_normal
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.dim_reduction.projection import FPCA

import warnings
warnings.filterwarnings('ignore')


def calc_sigma(x1, x2, l=1):
    """
    Compute the covariance matrix using the Gaussian kernel.

    Parameters
    x1 : array-like
    x2 : array-like
    l : Length-scale parameter

    Returns
    Sigma : The covariance matrix computed between `x1` and `x2`.
    """

    Sigma = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            Sigma[i, j] = np.exp(-0.5 * (np.abs(x1[i] - x2[j]) / l) ** 2)

    return Sigma


def gen_sim1(mean_func1, mean_func2, sigma1, sigma2, n1, n2, r, t):
    """
    (Simulation 1)
    Generate simulated functional samples using Gaussian process.

    Parameters
    mean_func1 : Mean functions of group 1.
    mean_func2 : Mean functions of group 2.
    sigma1 : Noise of group 1.
    sigma2 : Noise of group 2.
    n1 : Number of samples to generate for group 1.
    n2 : Number of samples to generate for group 2.
    r : Number of functional data per sample.
    t : Number of time points.

    Returns
    X : Simulated functional data for all subjects.
    y : Labels: 1 for group 1, 0 for group 2.
    """

    n = n1 + n2
    x = np.linspace(0, t-1, t)
    x_star = np.linspace(0, t-1, t)
    
    k_xx = calc_sigma(x, x)
    k_xxs = calc_sigma(x, x_star)
    k_xsx = calc_sigma(x_star, x)
    k_xsxs = calc_sigma(x_star, x_star)
    k_xx_inv1 = solve(k_xx + sigma1**2 * np.eye(len(k_xx)), np.eye(len(k_xx)))
    k_xx_inv2 = solve(k_xx + sigma2**2 * np.eye(len(k_xx)), np.eye(len(k_xx)))

    mu1 = k_xsx @ k_xx_inv1 @ mean_func1.T
    mu2 = k_xsx @ k_xx_inv2 @ mean_func2.T
    cov1 = k_xsxs - k_xsx @ k_xx_inv1 @ k_xxs
    cov2 = k_xsxs - k_xsx @ k_xx_inv2 @ k_xxs

    X = np.zeros((n, t, r))
        
    for j in range(r):
        X[:n1, :, j] = multivariate_normal.rvs(mean = mu1.iloc[:, j], cov = cov1, size = n1)
        X[n1:, :, j] = multivariate_normal.rvs(mean = mu2.iloc[:, j], cov = cov2, size = n2)

    y = [1] * n1 + [0] * n2            
    
    return X, y


def gen_fpca_samples(data, n_samples, n_components=50):
    """
    Generate synthetic functional samples using FPCA.

    Parameters
    data : Functional data where each row is a sample and each column is a time point.
    n_samples : Number of samples to generate.
    n_components : Number of principal components.

    Returns
    samples : Simulated functional samples of shape (n_samples, time_points).
    """

    time_points = data.shape[1]
    grid = np.linspace(0, 1, time_points)
    fd = FDataGrid(data_matrix=data.values, grid_points=grid)

    fpca = FPCA(n_components=n_components)
    fpca.fit(fd)

    mean_function = fpca.mean_.data_matrix[0].flatten()
    eigen_functions = np.array([ef.data_matrix[0, :, 0] for ef in fpca.components_])
    eigen_values = fpca.explained_variance_
    
    ksi = np.random.normal(0, np.sqrt(eigen_values), (n_samples, len(eigen_values)))
    samples = mean_function.values.flatten() + np.dot(ksi, eigen_functions)

    return samples


def gen_sim2(data1, data2, n1, n2, r):
    """
    (Simulation 2)
    Generate simulated functional samples using FPCA.

    Parameters
    data1 : Functional data of group 1.
    data2 : Functional data for group 2.
    n1 : Number of samples to generate for group 1.
    n2 : Number of samples to generate for group 2.
    r : Number of functional data per sample.

    Returns
    X : Simulated functional data for all subjects.
    y : Labels: 1 for group 1, 0 for group 2.
    """

    n = n1 + n2
    t = data1[0].shape[1]

    X = np.zeros((n, t, r))
        
    for j in range(r):

        df1 = pd.DataFrame([df.iloc[j] for df in data1])
        df2 = pd.DataFrame([df.iloc[j] for df in data2])

        X[:n1, :, j] = gen_fpca_samples(df1, n1)
        X[n1:, :, j] = gen_fpca_samples(df2, n2)

    y = [1]*n1 + [0]*n2            
    
    return X, y





