#include "functions.h"

typedef int (*DeterminantType)(double *A, __CLPK_integer k, __CLPK_integer lda, double *res);
typedef void (*InverseType)(double* A, int N);
typedef void (*EigenType)(int N, double *A, double *wr, double *vr);
typedef void (*SVDType)(int M, int N, double *A, double *s, double *u, double *vtt);

int c_determinant(double *A, __CLPK_integer k, __CLPK_integer lda, double *res);
void c_inverse( double* A, int N );
void c_eigen(int N, double *A, double *wr, double *vr);
void c_svd(int M, int N, double *A, double *s, double *u, double *vtt);
