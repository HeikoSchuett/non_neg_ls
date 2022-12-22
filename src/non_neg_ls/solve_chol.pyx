# cython language_level=3
import cython
from libc.math cimport sqrt

# Comments to keep track of:
# - for now perm is only kept for the nonzero entries



@cython.boundscheck(False)
@cython.nogil
@cython.cdivision(True)
@cython.wraparound(False)
cpdef add_rc(double [:,:] L, int [:] perm, int n, int p, double [:] Acol):
    """ adds a new row/column to the decomposition """
    cdef int i, ip
    cdef float sum
    perm[n] = p
    if n > 0:
        solve_lower(L, perm, n, Acol[:n], L[p, :n])
    sum = Acol[n]
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
        sum = y[i]
        for k in range(i):
            kp = perm[k]
            sum -= L[ip, kp] * x[kp]
        x[i] = sum / L[ip, ip]
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
        sum = y[i]
        for k in range(n-1, i, -1):
            kp = perm[k]
            sum -= L[kp, ip] * x[kp]
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

