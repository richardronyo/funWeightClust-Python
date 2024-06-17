#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "functions.h"
#include "mixture_wrapper.h"


int main() {
    int N = 3; // Example value for N
    int p = 2; // Example value for p
    int q = 2; // Example value for q
    int G = 1; // Example value for G
    int g = 0; // Example value for g

    // Example input arrays
    double x[] = {1.0, 2.0,
                  3.0, 4.0,
                  5.0, 6.0}; // N x p matrix
    double y[] = {2.0, 1.0,
                  4.0, 3.0,
                  6.0, 5.0}; // N x q matrix
    double gam[] = {1.0, 0.0,
                    0.0, 1.0}; // q x p matrix
    double cov[] = {1.0, 0.0,
                    0.0, 1.0}; // q x q matrix

    // Result array
    double delta[N];

    // Call the function
    c_C_rmahalanobis(&N, &p, &q, &G, &g, x, y, gam, cov, delta);

    // Print the results
    printf("Result:\n");
    for(int i = 0; i < N; i++) {
        printf("%f\n", delta[i]);
    }

    return 0;
}