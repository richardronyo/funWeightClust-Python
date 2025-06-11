#ifndef FUNCLUSTWEIGHT_H
#define FUNCLUSTWEIGHT_H

void C_mstep(char** modely, int *NN, int *pp, int* qq, int *GG, double *pi, double *x, double *y, double *t, double *gami, double *covyi, double *icovyi, double *logi, double *mtol, int *mmax);
void C_rmahalanobis(int *NN, int *pp, int *qq, int *GG, int *gg, double *x, double *y, double *gam, double *cov, double *delta);
void C_imahalanobis(double * x, double * muk, double * wk, double * Qk, double * aki, int * pp, int * pN, int * pdi, double *res);

#endif
