from scipy.optimize import nnls
from fnnls import fnnls
import numpy as np
from time import time
from matplotlib import pyplot as plt
from non_neg_ls import nn_least_squares


def run_test(Ns=[100, 500], method='scipy'):
    times = np.zeros_like(Ns, dtype=float)
    for iN, N in enumerate(Ns):
        A = np.random.rand(3 * N, N)
        y = np.ones(3 * N)
        t0 = time()
        if method == 'scipy':
            x = nnls(A, y)
        elif method == 'fnnls':
            x = fnnls(A, y)
        elif method == 'chol':
            x = nn_least_squares(A, y)
        elif method == 'ols':
            x = np.linalg.solve(A.T@A, A.T @ y)
        times[iN] = time() - t0
    return times


def make_plot(Ns=np.arange(10, 100, 10)):
    """ basic timing plot
    """
    methods = ['scipy', 'fnnls', 'chol', 'ols']
    times = np.zeros((len(methods), len(Ns)))
    for im, m in enumerate(methods):
        times[im] = run_test(Ns, m)
    plt.loglog(Ns, times)
