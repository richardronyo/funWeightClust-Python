#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imahalanobis_wrapper.h"
#include "imahalanobis.h"

void load_aki(double aki[], int r, int c)
{
    FILE *file;
    char *filename = "../data/aki.csv";
    char buffer[1024];

    file = fopen(filename, "r");
    int current_row = 0;

    char *token = fgets(buffer, sizeof(buffer), file);

    while (current_row < r)
    {
        char *leading_entry = strtok(token, ",");
        aki[current_row*c] = atof(leading_entry);

        for (int i = 1; i < r; i++)
        {
            char *next_entry = strtok(NULL, ",");
            aki[current_row*c + i] = atof(next_entry);

        }
        token = fgets(buffer, sizeof(buffer), file);
        current_row++;
    }

    fclose(file);
}

void load_qk(double qk[], int r, int c)
{
    FILE *file;
    char *filename = "../data/qk.csv";
    char buffer[1024];

    file = fopen(filename, "r");
    int current_row = 0;

    char *token = fgets(buffer, sizeof(buffer), file);

    while (current_row < r)
    {
        char *leading_entry = strtok(token, ",");
        qk[current_row*c] = atof(leading_entry);

        for (int i = 1; i < r; i++)
        {
            char *next_entry = strtok(NULL, ",");
            qk[current_row*c + i] = atof(next_entry);

        }
        token = fgets(buffer, sizeof(buffer), file);
        current_row++;
    }

    fclose(file);
}


void load_wk(double wk[], int r, int c)
{
    FILE *file;
    char *filename = "../data/wk.csv";
    char buffer[1024];

    file = fopen(filename, "r");
    int current_row = 0;

    char *token = fgets(buffer, sizeof(buffer), file);

    while (current_row < r)
    {
        char *leading_entry = strtok(token, ",");
        wk[current_row*c] = atof(leading_entry);

        for (int i = 1; i < r; i++)
        {
            char *next_entry = strtok(NULL, ",");
            wk[current_row*c + i] = atof(next_entry);

        }
        token = fgets(buffer, sizeof(buffer), file);
        current_row++;
    }

    fclose(file);
}

void load_muk(double muk[], int r)
{
    FILE *file;
    char *filename = "../data/muki.csv";
    char buffer[1024];

    file = fopen(filename, "r");
    int current_row = 0;

    char *token = fgets(buffer, sizeof(buffer), file);

    while (current_row < r)
    {
        muk[current_row] = atof(token);

        token = fgets(buffer, sizeof(buffer), file);
        current_row++;
    }

    fclose(file);
}
void load_x(int basis, double x[], int r, int c) {
    FILE *file;
    char *x_filename;
    char buffer[1024];

    if (basis == 30) {
        x_filename = "../data/cingulum_xf30.csv";
    } else if (basis == 40) {
        x_filename = "../data/cingulum_x_bs40.csv";
    } else if (basis == 20) {
        x_filename = "../data/cingulum_xf20.csv";
    } else {
        fprintf(stderr, "Invalid basis value.\n");
        exit(EXIT_FAILURE);
    }

    // Load X data
    file = fopen(x_filename, "r");
    int current_row = 0;
    fgets(buffer, sizeof(buffer), file);
    char *token = fgets(buffer, sizeof(buffer), file);
    while(current_row < r)
    {
        char *leading_entry = strtok(token, ",");
        x[current_row*c] = atof(leading_entry);
        for (int i = 1; i < c; i++)
        {
            char *next_entry = strtok(NULL, ",");
            x[current_row*c + i] = atof(next_entry);
        }
        token = fgets(buffer, sizeof(buffer), file);
        current_row++;
    }
    fclose(file);

}

void t(double matrix[], double transpose[], int r, int c)
{
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            transpose[j*r + i] = matrix[i*c + j];
        }
    }
}


int main() {
    int basis = 30;  // Example basis value

    int pN = 89, pp = 31, pdi = 31;
    double x[pN*pp], new_x[pN*pp];
    double muk[pp];
    double wk[pp*pdi];
    double qk[pp*pp];
    double aki[pp*pp];

    load_x(basis, x, pN, pp);
    t(x, new_x, pN, pp);
    print_mat("Transposed X:", new_x, pN, pp);
    
    load_muk(muk, pp);
    load_wk(wk, pp, pdi);
    load_qk(qk, pp, pp);
    load_aki(aki, pp, pp);

    double res[pN], new_res[pN];

    for (int i = 0; i < pN; i++)
    {
        res[i] = 0;
        new_res[i] = 0;
    }

    double * Qi = (double*)malloc(sizeof(double)*(pp*pdi));
    double * xQi = (double*)malloc(sizeof(double)*(pN*pdi));
    double * proj = (double*)malloc(sizeof(double)*(pN*pdi));
    
    load_muk(muk, pp);
    load_wk(wk, pp, pdi);
    load_qk(qk, pp, pp);
    load_aki(aki, pp, pp);


    printf("\n\nX - MUK:\n\n");
    for (int i = 0; i < pN; i++)
    {
        printf("Row %d:\t%f\t", i);
        for (int j = 0; j < pp; j++)
        {
            new_x[RC2IDX(i, j, pN)] = new_x[RC2IDX(i, j, pN)] - muk[j];   
            printf("%f\t", new_x[RC2IDX(i, j, pN)]);    
        }
        printf("\n");
    }
    

    return 0;
}