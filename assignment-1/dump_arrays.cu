
#include <stdlib.h>
#include <stdio.h>


int main(int argc,char **argv) {
    printf("Usage ./dump_arrays.out size array_file1 array_file2");

    int N =pow(2,atoi(argv[1]));
    size_t size = N * sizeof(float);
    int loop;

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    FILE *arrayfile_a;
    FILE *arrayfile_b;

    if (argc<3){
        printf("Too few arguments.\nUsage is ./ee16b068_3.out file1.txt file2.txt ");
        return 1;
    }

    // Initialize input vectors
    arrayfile_a = fopen(argv[2], "w");
    arrayfile_b = fopen(argv[3], "w");

    printf("\n Array A (first 10 values) \n ");
    for (loop = 0; loop < N; loop++)
    {
        h_A[loop] = rand() % 100 + 1;
        
        if (loop<10){
            printf("%f ", h_A[loop]);
        }

        fprintf(arrayfile_a, "%f\n", h_A[loop]);
    }

    printf("\n Array B (first 10 values) \n ");
    for (loop = 0; loop < N; loop++)
    {
        h_B[loop] = rand() % 100 + 1;
        
        if (loop<10){
            printf("%f ", h_B[loop]);
        }
        fprintf(arrayfile_b, "%f\n", h_B[loop]);
    }
    
    // Free host memory
    free(arrayfile_a); free(arrayfile_b);
    free(h_A);
    free(h_B);
    return 0;
}