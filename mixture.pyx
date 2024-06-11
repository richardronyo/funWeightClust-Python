from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np

cdef extern from "src/mixture_wrapper.h":
    int c_determinant(double *A, int k, int lda, double *res)

cdef extern from "src/mixture_wrapper.h":
    void c_inverse(double *A, int N)

cdef extern from "src/mixture_wrapper.h":
    void c_eigen(int N, double *A, double *wr, double *vr)

cdef extern from "src/mixture_wrapper.h":
    void c_svd(int M, int N, double *A, double *s, double *u, double *vtt)

def determinant(np.ndarray[np.float64_t, ndim=2] A):
    cdef int k = A.shape[0]
    cdef int lda = A.shape[1]
    cdef double res
    
    A = np.ascontiguousarray(np.transpose(A), dtype=np.float64)

    info = c_determinant(<double*> A.data, k, lda, &res)

    return res

def inverse(np.ndarray[np.float64_t, ndim=2] A):
    cdef int N = A.shape[0]
    
    A = np.ascontiguousarray(np.transpose(A), dtype=np.float64)

    c_inverse(<double*> A.data, N)

    return np.transpose(A)

def eigen(np.ndarray[np.float64_t, ndim=2] A):
    cdef int N = <int>A.shape[0]

    A = np.ascontiguousarray(np.transpose(A), dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] wr = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] vr = np.zeros((N, N), dtype=np.float64)

    c_eigen(N, <double*>A.data, <double*>wr.data, <double*>vr.data)

    return wr, vr.T

def svd(np.ndarray[np.float64_t, ndim=2] A):
    cdef int M = <int>A.shape[0]
    cdef int N = <int>A.shape[1]

    A = np.ascontiguousarray(np.transpose(A), dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] s = np.zeros(N, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] u = np.zeros((M, M), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] vtt = np.zeros((N, N), dtype=np.float64)

    c_svd(M, N, <double*> np.PyArray_DATA(A), <double*> np.PyArray_DATA(s), <double*> np.PyArray_DATA(u), <double*>np.PyArray_DATA(vtt))

    return s, u, vtt



