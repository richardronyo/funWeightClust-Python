#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "mixture_wrapper.h"


void print_mat(const char *name, double *matrix, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%10.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void c_C_mstep(char** modely, int *NN, int *pp, int* qq, int *GG,double *pi, double *x, double *y, double *t, double *gami ,double *covyi,double *icovyi,double *logi,double *mtol, int *mmax)
{
    HMODULE hModule = LoadLibrary("funclustweight.dll");
    if (hModule == NULL) {
        printf("Failed to load DLL\n");
        return;
    }

    mstepType C_mstep = (mstepType)GetProcAddress(hModule, "C_mstep");
    if (!C_mstep) {
        fprintf(stderr, "Failed to get C_mstep function address\n");
        FreeLibrary(hModule);
        return;
    }
    C_mstep(modely, NN, pp, qq, GG, pi, x, y, t, gami, covyi, icovyi, logi, mtol, mmax);
}


void c_C_rmahalanobis(int *NN, int *pp,int *qq,int *GG, int *gg, double *x,double *y, double *gam, double *cov, double *delta)
{
    HMODULE hModule = LoadLibrary("funclustweight.dll");
    if (hModule == NULL) 
    {
        printf("Failed to load DLL\n");
        return;
    }
    
    C_rmahalanobisType C_rmahalanobis = (C_rmahalanobisType)GetProcAddress(hModule, "C_rmahalanobis");
    if (!C_rmahalanobis) {
        fprintf(stderr, "Failed to get C_mstep function address\n");
        FreeLibrary(hModule);
        return;
    }
    C_rmahalanobis(NN, pp, qq, GG, gg, x, y, gam, cov, delta);
}

