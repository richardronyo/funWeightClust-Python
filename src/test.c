#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "functions.h"
#include "mixture_wrapper.h"

int main() {
    int N = 3;
    int p = 2;
    int q = 2;
    int G = 1;
    int g = 0;

    double x[6] = {1, 2, 3, 4, 5, 6}; // 3x2 matrix
    double y[6] = {2, 4, 6, 8, 10, 12}; // 3x2 matrix
    double gam[4] = {1, 0, 0, 1}; // 2x2 identity matrix
    double cov[4] = {1, 0, 0, 1}; // 2x2 identity matrix
    double delta[3] = {0, 0, 0}; // Result array

    c_C_rmahalanobis(&N, &p, &q, &G, &g, x, y, gam, cov, delta);

    printf("Delta values:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", delta[i]);
    }

    return 0;
}
