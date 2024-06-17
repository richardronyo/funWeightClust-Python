#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "mixture_wrapper.h"

int c_determinant(double *A, __CLPK_integer k, __CLPK_integer lda, double *res)
{
    HMODULE hModule = LoadLibrary("funclustweight.dll");
    DeterminantType determinant = (DeterminantType)GetProcAddress(hModule, "determinant");
    int info = determinant(A, k, lda, res);

    return info;
}

void c_inverse(double *A, int N)
{
    HMODULE hModule = LoadLibrary("funclustweight.dll");
    InverseType inverse = (InverseType)GetProcAddress(hModule, "inverse");
    inverse(A, N);
}

void c_eigen(int N, double *A, double *wr, double *vr)
{
    HMODULE hModule = LoadLibrary("funclustweight.dll");
    EigenType eigen = (EigenType)GetProcAddress(hModule, "eigen");
    eigen(N, A, wr, vr);
}

void c_svd(int M, int N, double *A, double *s, double *u, double *vtt)
{
    HMODULE hModule = LoadLibrary("funclustweight.dll");
    SVDType svd = (SVDType)GetProcAddress(hModule, "svd");
    svd(M, N, A, s, u, vtt);
}

void c_Gam1(int *NN, int *pp, int *qq,int *GG, double *x,double *y, double *z, int *gg, double *gam)
{
    HMODULE hModule = LoadLibrary("funclustweight.dll");
    Gam1Type gam1 = (Gam1Type)GetProcAddress(hModule, "Gam1");
    gam1(NN, pp, qq, GG, x, y, z, gg, gam);
}

void c_CovarianceY(int *NN, int *pp, int *qq,int *GG, double *x,double *y, double *z,double *gam, int *gg, double *Sigma)
{
    HMODULE hModule = LoadLibrary("funclustweight.dll");
    CovarianceYType covy = (CovarianceYType)GetProcAddress(hModule, "CovarianceY");
    covy(NN, pp, qq, GG, x, y, z, gam, gg, Sigma);
}

void c_C_mstep(char** modely, int *NN, int *pp, int* qq, int *GG,double *pi, double *x, double *y, double *t, double *gami ,double *covyi,double *icovyi,double *logi,double *mtol, int *mmax)
{
    HMODULE hModule = LoadLibrary("funclustweight.dll");
    C_mstepType c_mstep = (C_mstepType)GetProcAddress(hModule, "C_mstep");
    c_mstep(modely, NN, pp, qq, GG, pi, x, y, t, gami, covyi, icovyi, logi, mtol, mmax);
}

void c_C_rmahalanobis(int *NN, int *pp,int *qq,int *GG, int *gg, double *x,double *y, double *gam, double *cov, double *delta)
{
    HMODULE hModule = LoadLibrary("funclustweight.dll");
    C_rmahalanobisType C_rmahalanobis = (C_rmahalanobisType)GetProcAddress(hModule, "C_rmahalanobis");
    C_rmahalanobis(NN, pp, qq, GG, gg, x, y, gam, cov, delta);
}