from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np

cdef extern from "src/imahalanobis_wrapper.h":
    void c_C_imahalanobis(double * x, double * muk, double * wk, double * Qk, double * aki, int * pp, int * pN, int * pdi, double *res);

def C_imahalanobis(np.ndarray[np.float64_t, ndim=2]x, np.ndarray[np.float64_t, ndim=1]muk, np.ndarray[np.float64_t, ndim=2]wk, np.ndarray[np.float64_t, ndim=2]Qk, np.ndarray[np.float64_t, ndim=2]aki, int pp, int pN, int pdi, np.ndarray[np.float64_t, ndim=1]res):
    
    cdef int pp_val = pp
    cdef int pN_val = pN
    cdef int pdi_val = pdi

    cdef np.ndarray[np.float64_t, ndim=2] new_x = np.ascontiguousarray(np.transpose(x))
    cdef np.ndarray[np.float64_t, ndim=1] new_muk = np.ascontiguousarray(np.transpose(muk))
    cdef np.ndarray[np.float64_t, ndim=2] new_wk = np.ascontiguousarray(np.transpose(wk))
    cdef np.ndarray[np.float64_t, ndim=2] new_Qk = np.ascontiguousarray(np.transpose(Qk))
    cdef np.ndarray[np.float64_t, ndim=2] new_aki = np.ascontiguousarray(np.transpose(aki))
    cdef np.ndarray[np.float64_t, ndim=1] new_res = np.ascontiguousarray(np.transpose(res))

    c_C_imahalanobis(<double*>new_x.data, <double*>new_muk.data, <double*>new_wk.data, <double*>new_Qk.data, <double*>new_aki.data, &pp_val, &pN_val, &pdi_val, <double*>new_res.data)

    return new_res
    