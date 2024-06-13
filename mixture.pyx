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

cdef extern from "src/mixture_wrapper.h":
    void c_Gam1(int *NN, int *pp, int *qq,int *GG, double *x,double *y, double *z, int *gg, double *gam);
    
cdef extern from "src/mixture_wrapper.h":
    void c_CovarianceY(int *NN, int *pp, int *qq,int *GG, double *x,double *y, double *z,double *gam, int *gg, double *Sigma);

cdef extern from "src/mixture_wrapper.h":
    void c_C_mstep(char** modely, int *NN, int *pp, int* qq, int *GG,double *pi, double *x, double *y, double *t, double *gami ,double *covyi,double *icovyi,double *logi,double *mtol, int *mmax);

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


def Gam1(int NN, int pp, int qq, int GG, np.ndarray[np.float64_t, ndim=2] x, np.ndarray[np.float64_t, ndim=2] y, np.ndarray[np.float64_t, ndim=2] z, int gg, np.ndarray[np.float64_t, ndim=2] gam):
    cdef int NN_val  = <int>NN
    cdef int pp_val = <int>pp
    cdef int qq_val = <int>qq
    cdef int GG_val = <int>GG
    cdef int gg_val = <int>gg
    
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray(z)
    gam = np.ascontiguousarray(gam)
    c_Gam1(&NN_val, &pp_val, &qq_val, &GG_val, <double*>x.data, <double*>y.data, <double*>z.data, &gg_val, <double*>gam.data)


def CovarianceY(int NN, int pp, int qq, int GG, np.ndarray[np.float64_t, ndim=2] x, np.ndarray[np.float64_t, ndim=2] y, np.ndarray[np.float64_t, ndim=2] z, np.ndarray[np.float64_t, ndim=2] gam, int gg, np.ndarray[np.float64_t, ndim=2] Sigma):
    cdef int NN_val  = <int>NN
    cdef int pp_val = <int>pp
    cdef int qq_val = <int>qq
    cdef int GG_val = <int>GG
    cdef int gg_val = <int>gg

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray(z)
    gam = np.ascontiguousarray(gam)
    Sigma = np.ascontiguousarray(Sigma)

    c_CovarianceY(&NN_val, &pp_val, &qq_val, &GG_val, <double*>x.data, <double*>y.data, <double*>z.data, <double*>gam.data, &gg_val, <double*>Sigma.data)

def C_mstep(str modely, int NN, int pp, int qq, int GG, np.ndarray[np.float64_t, ndim=1] pi, np.ndarray[np.float64_t, ndim=2] x, np.ndarray[np.float64_t, ndim=2] y, np.ndarray[np.float64_t, ndim=2] t, np.ndarray[np.float64_t, ndim=2] gami, np.ndarray[np.float64_t, ndim=2] covyi, np.ndarray[np.float64_t, ndim=2] icovyi, np.ndarray[np.float64_t, ndim=1] logi, float mtol, int mmax):
    cdef int NN_val  = <int>NN
    cdef int pp_val = <int>pp
    cdef int qq_val = <int>qq
    cdef int GG_val = <int>GG
    cdef int mmax_val = <int>mmax

    cdef double mtol_val = <double>mtol

    byte_modely = modely.encode('utf-8')
    cdef char* modely_cstr = byte_modely

    c_C_mstep(&modely_cstr, &NN_val, &pp_val, &qq_val, &GG_val, <double*>pi.data, <double*>x.data, <double*>y.data, <double*>t.data, <double*>gami.data, <double*>covyi.data, <double*>icovyi.data, <double*>logi.data, &mtol_val, &mmax_val)

    return (gami, covyi, icovyi, logi)