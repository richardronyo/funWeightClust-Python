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