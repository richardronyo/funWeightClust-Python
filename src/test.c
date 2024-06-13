#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "functions.h"
#include "mixture_wrapper.h"

int main() {
    int NN = 10;
    int pp = 3;
    int qq = 2;
    int GG = 2;
    int mmax = 100;
    double mtol = 1e-40;

    // Model type (example)
    char* modely[1] = {"VII"};

    // Mixing proportions
    double pi[2] = {0.5, 0.5};

    // Predictor matrix (N x p)
    double x[10][3] = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9},
        {1.0, 1.1, 1.2},
        {1.3, 1.4, 1.5},
        {1.6, 1.7, 1.8},
        {1.9, 2.0, 2.1},
        {2.2, 2.3, 2.4},
        {2.5, 2.6, 2.7},
        {2.8, 2.9, 3.0}
    };

    // Response matrix (N x q)
    double y[10][2] = {
        {0.1, 0.2},
        {0.3, 0.4},
        {0.5, 0.6},
        {0.7, 0.8},
        {0.9, 1.0},
        {1.1, 1.2},
        {1.3, 1.4},
        {1.5, 1.6},
        {1.7, 1.8},
        {1.9, 2.0}
    };

    // Weights matrix (N x G)
    double t[10][2] = {
        {0.1, 0.2},
        {0.3, 0.4},
        {0.5, 0.6},
        {0.7, 0.8},
        {0.9, 1.0},
        {1.1, 1.2},
        {1.3, 1.4},
        {1.5, 1.6},
        {1.7, 1.8},
        {1.9, 2.0}
    };

    // Allocate memory for output arrays
    double gami[2][6] = {0}; // q * p
    double covyi[2][4] = {0}; // q * q
    double icovyi[2][4] = {0}; // q * q
    double logi[2] = {0};

    // Call the C_mstep function
    c_C_mstep(modely, &NN, &pp, &qq, &GG, pi, (double*)x, (double*)y, (double*)t, (double*)gami, (double*)covyi, (double*)icovyi, logi, &mtol, &mmax);

    // Print the resulting matrices
    printf("gami matrix:\n");
    for (int g = 0; g < GG; g++) {
        for (int i = 0; i < qq; i++) {
            for (int j = 0; j < pp; j++) {
                printf("%.30f ", gami[g*qq*pp + i*pp + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("covyi matrix:\n");
    for (int g = 0; g < GG; g++) {
        for (int i = 0; i < qq; i++) {
            for (int j = 0; j < qq; j++) {
                printf("%.30f ", covyi[g*qq*qq + i*qq + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("icovyi matrix:\n");
    for (int g = 0; g < GG; g++) {
        for (int i = 0; i < qq; i++) {
            for (int j = 0; j < qq; j++) {
                printf("%.30f ", icovyi[g*qq*qq + i*qq + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("logi array:\n");
    for (int g = 0; g < GG; g++) {
        printf("%f ", logi[g]);
    }
    printf("\n");

    return 0;
}