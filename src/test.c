#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "functions.h"
#include "mixture_wrapper.h"

int main() {


    __CLPK_integer k = 3;
    __CLPK_integer lda = 3;
    double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 8.0, 9.0};
    double res;

    int info = c_determinant(A, k, lda, &res);

    printf("Determinant: %lf\n", res);
    
   
    return 0;
}
