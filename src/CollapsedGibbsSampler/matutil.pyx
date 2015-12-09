cimport cython
'''
Normalization with increment mu
'''
@cython.boundscheck(False)
cpdef void normalize(double[:, ::1] arr, double mu) nogil:
    cdef double cumulative_p
    cdef size_t i, j
    for i in xrange(arr.shape[0]):
        cumulative_p = 0
        for j in xrange(arr.shape[1]):
            arr[i,j] += mu
            cumulative_p += arr[i,j]
        for j in xrange(arr.shape[1]):
            arr[i,j] /= cumulative_p
            
           
'''
Array methods in cython
'''
def copy1d(int[::1] sum_K, int[::1] t_sum_K):
    with nogil:
        _copy1d(sum_K, t_sum_K)

def subtract1d(int[::1] sum_K, int[::1] t_sum_K):
    with nogil:
        _subtract1d(sum_K, t_sum_K)
        
def add1d(int[::1] sum_K, int[::1] t_sum_K):
    with nogil:
        _add1d(sum_K, t_sum_K)
        
def copy2d(double[:, ::1] K_V, double[:, ::1] t_K_V):
    with nogil:
        _copy2d(K_V, t_K_V)
        
def subtract2d(double[:, ::1] K_V, double[:, ::1] t_K_V):
    with nogil:
        _subtract2d(K_V, t_K_V)
        
def add2d(double[:, ::1] K_V, double[:, ::1] t_K_V):
    with nogil:
        _add2d(K_V, t_K_V)

@cython.boundscheck(False)
cdef inline void _copy1d(int[::1] sum_K, int[::1] t_sum_K) nogil:
    cdef int i
    for i in xrange(sum_K.shape[0]):
        t_sum_K[i] = sum_K[i]
        
@cython.boundscheck(False)
cdef inline void _subtract1d(int[::1] t_sum_K, int[::1] sum_K) nogil:
    cdef int i
    for i in xrange(sum_K.shape[0]):
        t_sum_K[i] -= sum_K[i]
        
@cython.boundscheck(False)
cdef inline void _add1d(int[::1] sum_K, int[::1] t_sum_K) nogil:
    cdef int i
    for i in xrange(sum_K.shape[0]):
        sum_K[i] += t_sum_K[i]
        
@cython.boundscheck(False)
cdef inline void _copy2d(double[:, ::1] K_V, double[:, ::1] t_K_V) nogil:
    cdef int i, j
    for i in xrange(K_V.shape[0]):
        for j in xrange(K_V.shape[1]):
            t_K_V[i,j] = K_V[i,j]
        
@cython.boundscheck(False)
cdef inline void _subtract2d(double[:, ::1] t_K_V, double[:, ::1] K_V) nogil:
    cdef int i, j
    for i in xrange(K_V.shape[0]):
        for j in xrange(K_V.shape[1]):
            t_K_V[i,j] -= K_V[i,j]
        
@cython.boundscheck(False)
cdef inline void _add2d(double[:, ::1] K_V, double[:, ::1] t_K_V) nogil:
    cdef int i, j
    for i in xrange(K_V.shape[0]):
        for j in xrange(K_V.shape[1]):
            K_V[i,j] += t_K_V[i,j]