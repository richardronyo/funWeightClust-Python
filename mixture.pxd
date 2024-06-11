# mixture.pxd
cdef extern from "src/mixture_wrapper.h":
    int c_determinant(double *A, int k, int lda, double *res)
    void c_inverse(double *A, int N)
    void c_eigen(int N, double *A, double *wr, double *vr)
    void c_svd(int M, int N, double *A, double *s, double *u, double *vtt)



