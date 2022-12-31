# cython language_level=3
import numpy as np
import scipy.sparse.linalg
import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.math cimport sqrt
from libc.stdio cimport printf
cimport numpy as np
from scipy.linalg.blas import dgemm, dgemv
# cimport scipy.linalg.cython_blas
# cimport scipy.linalg.cython_lapack

# Comments to keep track of:
# - for now perm is only kept for the nonzero entries
# - in general all values outside of L[perm[:n], perm[:n]] are meaningless,
#   but not necessarily 0
# - also the unused half of the triangular matrix is not controlled
# - at the moment I am loading the blas functions via import not cimport


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def nn_least_squares(
    double [:, :] A,
    double [:] y,
    double ridge_weight=0,
    V=None,
    double eps=100 * np.finfo(float).eps):
    """ non-negative least squares
    essentially scipy.optimize.nnls extended to accept a ridge_regression
    regularisation and/or a covariance matrix V.

    The algorithm is discribed in detail here:
    Bro, R., & Jong, S. D. (1997). A fast non-negativity-constrained
    least squares algorithm. Journal of Chemometrics, 11, 9.

    This is an active set algorithm which is somewhat optimized by
    precomputing A^T V^-1 A and A^T V y such that during the optimization
    only matricies of rank r need to be inverted.

    Here, I further optimize by computing updating a cholesky decomposition
    of A^TA, L^T L, which is organized in a matrix of size A^TA and a 
    permutation vector perm.
    """
    cdef double [:] x, y_V_A, s_p, w
    cdef double [:, :] V_A, ATA, L
    cdef int n, i_alpha, i, n_max, i_max_w
    cdef int [:] perm
    cdef double alpha, alpha_test, max_w
    n_max = A.shape[1]
    x = np.zeros(n_max, 'float64')
    w = np.zeros(n_max, 'float64')
    s_p = np.zeros(n_max, 'float64')
    p = np.zeros(n_max, bool)
    ATA = np.empty((n_max, n_max), 'float64')
    # Initialize L
    L = np.empty((n_max, n_max), 'float64')
    perm = np.empty(n_max, 'int32')
    n = 0
    if V is None:
        # w = A.t @ y
        y_V_A = dgemv(1, A, y, 0, w, 0, 1, 0, 1, 1, 0)
        w = y_V_A.copy()
        # ATA = np.matmul(A.T, A) + ridge_weight * np.eye(A.shape[1])
        ATA = dgemm(1, A, A, 0, ATA, 1, 0, 1)
        for i in range(n_max):
            ATA[i, i] += ridge_weight 
    else:
        # not yet worked on!
        V_A = np.empty_like(A)
        for i in range(n_max):
            V_A[i] = scipy.sparse.linalg.cg(
                V, A[:, i], atol=10 ** -9)[0]
        y_V_A = np.matmul(V_A, y)
        w = y_V_A
        ATA = np.matmul(V_A, A) + ridge_weight * np.eye(A.shape[1])
    max_w = -1.0
    for i in range(n_max):
        if w[i] > max_w:
            max_w = w[i]
            i_max_w = i
    while max_w > 0:
        print('\n\n')
        print(max_w)
        # Update L
        add_rc(L, perm, n, i_max_w, ATA[i_max_w])
        n += 1
        print('n')
        print(n)
        print('ATA')
        print(np.array(ATA))
        print('L')
        print(np.array(L))
        print('perm')
        print(np.array(perm))
        # solve OLS
        print('start solve:')
        solve_lower(L, perm, n, y_V_A, s_p)
        print('s_p')
        print(np.array(s_p))
        solve_upper(L, perm, n, s_p, s_p)
        print('s_p')
        print(np.array(s_p))
        while smaller0(s_p, perm, n):
            alpha = 1
            i_alpha = -1
            for i in range(n):
                if s_p[perm[i]] < 0:
                    alpha_test = x[perm[i]] / (x[perm[i]] - s_p[perm[i]])
                    if alpha_test < alpha:
                        alpha = alpha_test
                        i_alpha = perm[i]
            # alphas = x[p] / (x[p] - s_p)
            # alphas[s_p > 0] = 1
            # i_alpha = np.argmin(alphas)
            # alpha = alphas[i_alpha]
            for i in range(n):
                x[perm[i]] = x[perm[i]] + alpha * (s_p[perm[i]] - x[perm[i]])
            # x[p] = x[p] + alpha * (s_p - x[p])
            # i_alpha = np.where(p)[0][i_alpha]
            x[perm[i_alpha]] = 0
            # remove entry from active set
            # p[perm[i_alpha] = False
            remove_rc(L, perm, n, i_alpha)
            n -= 1
            solve_lower(L, perm, n, y_V_A, s_p)
            solve_upper(L, perm, n, s_p, s_p)
        for i in range(n):
            x[perm[i]] = s_p[perm[i]]
        w = dgemv(1, ATA, x, 0, w, 0, 1, 0, 1, 1, 1)
        max_w = -1.0
        for i in range(n_max):
            w[i] = y_V_A[i] - w[i]
            if (x[i] == 0) and (w[i] > max_w):
                max_w = w[i]
                i_max_w = i
        print('w')
        print(np.array(w))
    """
    if V is None:
        loss = np.sum((y - A @ x) ** 2)
    else:
        loss = (y - A @ x).T @ V @ (y - A @ x)
    """
    return np.array(x) #, loss


@cython.boundscheck(False)
@cython.nogil
@cython.wraparound(False)
cdef (int, double) find_max(double [:] w, int n_max):
    cdef int i, i_max_w = 0
    cdef double max_w = -1.0
    for i in range(n_max):
        if w[i] > max_w:
            max_w = w[i]
            i_max_w = i
    return (i_max_w, max_w)


@cython.boundscheck(False)
@cython.nogil
@cython.wraparound(False)
cdef int smaller0(double [:] x, int [:] perm, int n):
    cdef int i
    for i in range(n):
        if x[perm[i]] < 0:
            return 1
    return 0

@cython.boundscheck(False)
@cython.nogil
@cython.cdivision(True)
@cython.wraparound(False)
cpdef remove_rc(double [:,:] L, int [:] perm, int n, int p):
    """removes a row/column from the decomposition"""
    cdef int ip, i
    ip = -1
    for i in range(n):
        if perm[i] == p:
            ip = i
            break
    if ip == -1:
        raise ValueError('The row/column to be removed is not part of L at the moment.')
    update_chol(L, perm[(ip+1):n], n - ip - 1, L[ip])
    for i in range(ip, n):
        perm[i] = perm[i+1]
    # n = n - 1


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nogil
cpdef add_rc(double [:,:] L, int [:] perm, int n, int p, double [:] Acol):
    """ adds a new row/column to the decomposition """
    cdef int i, ip
    cdef float sum
    perm[n] = p
    if n > 0:
        solve_lower(L, perm, n, Acol, L[p, :n])
    sum = Acol[p]
    for i in range(n):
        ip = perm[i]
        sum -= L[p, ip] * L[p, ip]
    L[p, p] = sqrt(sum)


@cython.boundscheck(False)
@cython.nogil
@cython.cdivision(True)
@cython.wraparound(False)
cpdef solve_lower(double [:,:] L, int [:] perm, int n, double [:] y, double [:] x):
    cdef int i, ip, k, kp
    cdef double sum
    for i in range(n):
        ip = perm[i]
        sum = y[ip]
        for k in range(i):
            kp = perm[k]
            sum -= L[ip, kp] * x[kp]
        x[ip] = sum / L[ip, ip]
    # return x


@cython.boundscheck(False)
@cython.nogil
@cython.cdivision(True)
@cython.wraparound(False)
cpdef solve_upper(double [:,:] L, int [:] perm, int n, double [:] y, double [:] x):
    cdef int i, ip, k, kp
    cdef double sum
    for i in range(n-1, -1, -1):
        ip = perm[i]
        sum = y[ip]
        for k in range(n-1, i, -1):
            kp = perm[k]
            sum -= L[kp, ip] * x[kp]
        x[ip] = sum / L[ip, ip]
    # return x


@cython.boundscheck(False)
@cython.nogil
@cython.cdivision(True)
@cython.wraparound(False)
cpdef update_chol(double [:,:] L, int [:] perm, int n, double [:] x):
    cdef int k, kp, i, ip
    cdef double r, c, s
    for kp in range(n-1):
        k = perm[kp]
        r = sqrt(L[k, k] * L[k, k] + x[k] * x[k])
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        for ip in range(k+1, n+1):
            i = perm[ip]
            L[i, k] = (L[i, k] + s * x[i]) / c
            x[i] = c * x[i] - s * L[i, k]
    k = perm[n]
    r = sqrt(L[k, k] * L[k, k] + x[k] * x[k])
    L[k, k] = r


@cython.boundscheck(False)
@cython.nogil
@cython.cdivision(True)
@cython.wraparound(False)
cpdef mult_LLT(double [:,:] L, int [:] perm, int n):
    """ For checking purposes this computes LL^T
    If the cholesky computations are correct this should give
    ATA[perm[:n], perm[:n]] at all times
    """
    cdef double [:,:] Result = np.zeros((n,n), 'float64')
    cdef int i, j, k, ip, jp, kp
    for i in range(n):
        ip = perm[i]
        for j in range(n):
            jp = perm[j]
            for k in range(n):
                kp = perm[k]
                Result[i, j] += L[ip, kp] * L[jp, kp]
                if k == i or k == j:
                    break
    return np.array(Result)
