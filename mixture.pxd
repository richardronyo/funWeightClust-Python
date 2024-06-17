# mixture.pxd
cdef extern from "src/mixture_wrapper.h":
    int c_determinant(double *A, int k, int lda, double *res)
    void c_inverse(double *A, int N)
    void c_eigen(int N, double *A, double *wr, double *vr)
    void c_svd(int M, int N, double *A, double *s, double *u, double *vtt)
    void c_Gam1(int *NN, int *pp, int *qq,int *GG, double *x,double *y, double *z, int *gg, double *gam)
    void c_CovarianceY(int *NN, int *pp, int *qq,int *GG, double *x,double *y, double *z,double *gam, int *gg, double *Sigma)
    void c_C_mstep(char** modely, int *NN, int *pp, int* qq, int *GG,double *pi, double *x, double *y, double *t, double *gami ,double *covyi,double *icovyi,double *logi,double *mtol, int *mmax)
    void c_C_rmahalanobis(int *NN, int *pp,int *qq,int *GG, int *gg, double *x,double *y, double *gam, double *cov, double *delta)


