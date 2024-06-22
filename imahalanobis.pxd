cdef extern from "src/imahalanobis_wrapper.h":
    void c_C_imahalanobis(double * x, double * muk, double * wk, double * Qk, double * aki, int * pp, int * pN, int * pdi, double *res);