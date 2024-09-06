cdef extern from "src/imahalanobis.h":
    void C_imahalanobis(double * x, double * muk, double * wk, double * Qk, double * aki, int * pp, int * pN, int * pdi, double *res);