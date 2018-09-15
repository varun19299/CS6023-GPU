#include <stdlib.h>
#include <stdio.h>

__global__ void VecAdd(float* A, float* B, float*
    C, int N){
    // Host code
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];

    }

int main(int argc,char **argv) {
    int N = pow(2,15);
    size_t size = N * sizeof(float);
    int loop;
    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    FILE *arrayfile_a;
    FILE *arrayfile_b;
    FILE *arrayfile_c;

    if (argc<2){
        printf("Too few arguments.\nUsage is ./ee16b068_3.out file1.txt file2.txt ");
        return 1;
    }

    // Initialize input vectors
    
    arrayfile_a = fopen(argv[1], "r");
    arrayfile_b = fopen(argv[2], "r");
    arrayfile_c = fopen("ee16b068_3_out.txt", "w");

    // Read first two arrays
    printf("\nArray A (first 10 values) \n ");
    for (loop = 0; loop < N; loop++)
    {
        fscanf(arrayfile_a, "%f", &h_A[loop]);
        if (loop<10){
            printf("%f ", h_A[loop]);
        }
    }

    printf("\nArray B (first 10 values) \n ");
    for (loop = 0; loop < N; loop++)
    {
        fscanf(arrayfile_b, "%f", &h_B[loop]);
        if (loop<10){
            printf("%f ", h_B[loop]);
        }
    }
    
    //printf("Array A (first 10 values) \n ");
    //for(loop = 0; loop < N; loop++){
    //h_A[loop] = rand() % 100 + 1;
    //if (loop<10){
    //    printf("%f ", h_A[loop]);
    //}
    //}

  /*   printf("\nArray B (first 10 values) \n ");
    for(loop = 0; loop < N; loop++){
    h_B[loop] = rand() % 100 + 1;
    if (loop<10){
        printf("%f ", h_B[loop]);
    }
    } */

    // Allocate vectors in device memory
    float* d_A;   cudaMalloc(&d_A, size);
    float* d_B;   cudaMalloc(&d_B, size);
    float* d_C;   cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size,cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) /threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B, d_C, N);

     // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size,cudaMemcpyDeviceToHost);

    printf("\nArray C (first 10 outputs)\n");
    for(loop = 0; loop < 10; loop++)
      printf("%f ", h_C[loop]);

    // Log outputs
    printf("\nWritting to file ee16b068_3_out.txt as <vec a> <vec b> <vec>");
    for (loop=0;loop<N;loop++){
        fprintf(arrayfile_c,"%f   %f   %f\n",h_A[loop],h_B[loop],h_C[loop]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(arrayfile_a); free(arrayfile_b);
    return 0;
}