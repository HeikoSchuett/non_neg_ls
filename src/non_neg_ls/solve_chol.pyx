# cython language_level=3
import cython
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.nogil
@cython.cdivision(True)
@cython.wraparound(False)
cpdef solve_lower(double [:,:] L, int [:] perm, int n, double [:] y, double [:] x):
    cdef int i, ip, k
    cdef double sum
    for i in range(n):
        ip = perm[i]
        sum = y[i]
        for k in range(i):
            sum -= L[ip, perm[k]] * x[k]
        x[i] = sum / L[ip, ip]
    # return x


@cython.boundscheck(False)
@cython.nogil
@cython.cdivision(True)
@cython.wraparound(False)
cpdef solve_upper(double [:,:] L, int [:] perm, int n, double [:] y, double [:] x):
    cdef int i, ip, k
    cdef double sum
    for i in range(n-1, -1, -1):
        ip = perm[i]
        sum = y[i]
        for k in range(n-1, i, -1):
            sum -= L[perm[k], ip] * x[k]
        x[i] = sum / L[ip, ip]
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

