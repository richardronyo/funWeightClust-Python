#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "imahalanobis_wrapper.h"

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


void c_C_imahalanobis(double * x, double * muk, double * wk, double * Qk, double * aki, int * pp, int * pN, int * pdi, double * res)
{
    HMODULE hModule = LoadLibrary("TFunHDDC.dll");
    if (hModule == NULL)
    {
        printf("Failed to load DLL \n");
        return;
    }

    C_imahalanobisType C_imahalanobis = (C_imahalanobisType)GetProcAddress(hModule, "C_imahalanobis");
    if (!C_imahalanobis) {
        fprintf(stderr, "Failed to get C_mstep function address\n");
        FreeLibrary(hModule);
        return;
    }
    C_imahalanobis(x, muk, wk, Qk, aki, pp, pN, pdi, res);
}