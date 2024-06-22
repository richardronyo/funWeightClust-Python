#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "imahalanobis_wrapper.h"

int main() {
    int p = 3, N = 4, di = 2;
    double x[12] = {1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0,
                    10.0, 11.0, 12.0};
    double muk[3] = {1.0, 2.0, 3.0};
    double wk[6] = {1.0, 0.0,
                    0.0, 1.0,
                    1.0, 1.0};
    double Qk[6] = {1.0, 0.5,
                    0.5, 1.0,
                    0.5, 0.5};
    double aki[4] = {1.0, 0.0,
                     0.0, 1.0};
    double res[4];

    c_C_imahalanobis(x, muk, wk, Qk, aki, &p, &N, &di, res);

    for (int i = 0; i < 4; i++)
    {
        printf("%f\t", res[i]);
    }

    printf("\n");
    

    return 0;
}
