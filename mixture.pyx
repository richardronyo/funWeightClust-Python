from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np

cdef extern from "src/mixture_wrapper.h":
    void c_C_mstep(char** modely, int *NN, int *pp, int* qq, int *GG, double *pi, double *x, double *y, double *t, double *gami, double *covyi, double *icovyi, double *logi, double *mtol, int *mmax)
    void c_C_rmahalanobis(int *NN, int *pp,int *qq,int *GG, int *gg, double *x,double *y, double *gam, double *cov, double *delta)

def C_mstep(str modely, int NN, int pp, int qq, int GG, pi, x, y, t, gami, covyi, icovyi, logi, float mtol, int mmax):
    """
    Description
    -----------
        The C_mstep function is a Cython wrapper for the C function c_C_mstep found in src/mixture_wrapper.h, which loads a .dll file that calls the C_mstep function found in the R API.

    Parameters
    -----------
        modely: (str)
            String that determines which y-model the code is supposed to use
        NN: (int)
            Number of rows in coefficient matrix x. Also the number of curves
        pp: (int)
            Number of columns in coefficient matrix x. Also the number of points per curves
        qq: (int)
            Number of columns in matrix y_ptr
        GG: (int)
            Number of clusters
        x: np.ndarray[np.float64_t, ndim=2], (N, p)
            Coefficient matrix
        y: np.ndarray[np.float64_t, ndim=2], (N, q)
            Coefficient matrix
        t: np.ndarray[np.float64_t, ndim=2], (N, K)
            Matrix that contains the probabilities of a curve belonging to one of K groups
        
        gami: np.ndarray[np.float64_t, ndim=2], (K, q*(p+1))
        covyi: np.ndarray[np.float64_t, ndim=2], (K, q*q)
        icovyi: np.ndarray[np.float64_t, ndim=2], (K, q*q)
        logi: np.ndarray[np.float64_t, ndim=1], (K)

    Returns
    -------
        gami: np.ndarray[np.float64_t, ndim=2], (K, q*(p+1))
        covyi: np.ndarray[np.float64_t, ndim=2], (K, q*q)
        icovyi: np.ndarray[np.float64_t, ndim=2], (K, q*q)
        logi: np.ndarray[np.float64_t, ndim=1], (K)

    """
    cdef int NN_val = NN
    cdef int pp_val = pp
    cdef int qq_val = qq
    cdef int GG_val = GG
    cdef int mmax_val = mmax

    cdef double mtol_val = mtol

    byte_modely = modely.encode('utf-8')
    cdef char* modely_cstr = byte_modely
    t = np.ascontiguousarray(t, dtype=np.float64)


    cdef np.ndarray[np.float64_t, ndim=1] new_pi = np.ascontiguousarray(np.transpose(pi))
    cdef np.ndarray[np.float64_t, ndim=2] new_x = np.ascontiguousarray(np.transpose(x))
    cdef np.ndarray[np.float64_t, ndim=2] new_y = np.ascontiguousarray(np.transpose(y))
    cdef np.ndarray[np.float64_t, ndim=2] new_t = np.ascontiguousarray(np.transpose(t))
    cdef np.ndarray[np.float64_t, ndim=2] new_gami = np.ascontiguousarray(np.transpose(gami))
    cdef np.ndarray[np.float64_t, ndim=2] new_covy = np.ascontiguousarray(np.transpose(covyi))
    cdef np.ndarray[np.float64_t, ndim=2] new_icovy = np.ascontiguousarray(np.transpose(icovyi))
    cdef np.ndarray[np.float64_t, ndim=1] new_logi = np.ascontiguousarray(np.transpose(logi))

    cdef double* pi_ptr = <double*>new_pi.data
    c_C_mstep(&modely_cstr, &NN_val, &pp_val, &qq_val, &GG_val, pi_ptr, <double*>new_x.data, <double*>new_y.data, <double*>new_t.data, <double*>new_gami.data, <double*>new_covy.data, <double*>new_icovy.data, <double*>new_logi.data, &mtol_val, &mmax_val)

    final_gami = new_gami.reshape((GG, qq, pp)).transpose(0, 2, 1)
    final_covyi = new_covy.reshape((GG, qq, qq)).transpose(0, 2, 1)
    final_icovyi = new_icovy.reshape(GG, qq, qq).transpose(0, 2, 1)
    return (final_gami, final_covyi, final_icovyi, logi)

def C_rmahalanobis(int NN, int pp, int qq, int GG, int gg, np.ndarray[np.float64_t, ndim=2] x, np.ndarray[np.float64_t, ndim=2] y, np.ndarray[np.float64_t, ndim=2] gam, np.ndarray[np.float64_t, ndim=2] cov, np.ndarray[np.float64_t, ndim=1] delta):
    """
    Description
    -----------
    The C_rmahalanobis function is a Cython wrapper for the c_C_rmahalanobis function found in src/mixture_wrapper.h, which loads a .dll file that calls the C_rmahalanobis function found in the R API.

    Parameters
    ----------
        NN: (int)
            Number of rows in coefficient matrix x. Also the number of curves
        pp: (int)
            Number of columns in coefficient matrix x. Also the number of points per curves
        qq: (int)
            Number of columns in matrix y_ptr
        GG: (int)
            Number of clusters
        x: np.ndarray[np.float64_t, ndim=2], (N, p)
            Coefficient matrix
        y: np.ndarray[np.float64_t, ndim=2], (N, q)
            Coefficient matrix
        t: np.ndarray[np.float64_t, ndim=2], (N, K)
            Matrix that contains the probabilities of a curve belonging to one of K groups

        gam: np.ndarray[np.float64_t, ndim=2] (K, q*(p+1))
        cov: np.ndarray[np.float64_t, ndim=2] (K, q*q)
        delta: (double)
            The mahalanobis distance between x and y.
    """
    # Ensure arrays are C-contiguous
    cdef np.ndarray[np.float64_t, ndim=2] new_x = np.ascontiguousarray(np.transpose(x))
    cdef np.ndarray[np.float64_t, ndim=2] new_y = np.ascontiguousarray(np.transpose(y))
    cdef np.ndarray[np.float64_t, ndim=2] new_gam = np.ascontiguousarray(np.transpose(gam))
    cdef np.ndarray[np.float64_t, ndim=2] new_cov = np.ascontiguousarray(np.transpose(cov))
    cdef np.ndarray[np.float64_t, ndim=1] final_delta = np.ascontiguousarray(np.transpose(delta))

    # Convert to 1D arrays
    cdef double* x_ptr = <double*>new_x.data
    cdef double* y_ptr = <double*>new_y.data
    cdef double* gam_ptr = <double*>new_gam.data
    cdef double* cov_ptr = <double*>new_cov.data
    cdef double* delta_ptr = <double*>final_delta.data
    c_C_rmahalanobis(&NN, &pp, &qq, &GG, &gg, x_ptr, y_ptr, gam_ptr, cov_ptr, delta_ptr)
    return final_delta