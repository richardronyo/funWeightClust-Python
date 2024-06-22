#include "imahalanobis.h"

typedef void (*C_imahalanobisType)(double * x, double * muk, double * wk, double * Qk, double * aki, int * pp, int * pN, int * pdi, double * res);

void c_C_imahalanobis(double * x, double * muk, double * wk, double * Qk, double * aki, int * pp, int * pN, int * pdi, double *res);
void print_mat(const char *name, double *matrix, int rows, int cols);