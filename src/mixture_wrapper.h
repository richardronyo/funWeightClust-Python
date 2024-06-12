#include "functions.h"

typedef int (*DeterminantType)(double *A, __CLPK_integer k, __CLPK_integer lda, double *res);
typedef void (*InverseType)(double* A, int N);
typedef void (*EigenType)(int N, double *A, double *wr, double *vr);
typedef void (*SVDType)(int M, int N, double *A, double *s, double *u, double *vtt);
typedef void (*Gam1Type)(int *NN, int *pp, int *qq,int *GG, double *x,double *y, double *z, int *gg, double *gam);
typedef void (*CovarianceYType)(int *NN, int *pp, int *qq,int *GG, double *x,double *y, double *z,double *gam, int *gg, double *Sigma);
typedef void (*C_mstepType)(char** modely, int *NN, int *pp, int* qq, int *GG,double *pi, double *x, double *y, double *t, double *gami ,double *covyi,double *icovyi,double *logi,double *mtol, int *mmax);

int c_determinant(double *A, __CLPK_integer k, __CLPK_integer lda, double *res);
void c_inverse( double* A, int N );
void c_eigen(int N, double *A, double *wr, double *vr);
void c_svd(int M, int N, double *A, double *s, double *u, double *vtt);
void c_Gam1(int *NN, int *pp, int *qq,int *GG, double *x,double *y, double *z, int *gg, double *gam);
void c_CovarianceY(int *NN, int *pp, int *qq,int *GG, double *x,double *y, double *z,double *gam, int *gg, double *Sigma);
void c_C_mstep(char** modely, int *NN, int *pp, int* qq, int *GG,double *pi, double *x, double *y, double *t, double *gami ,double *covyi,double *icovyi,double *logi,double *mtol, int *mmax);
