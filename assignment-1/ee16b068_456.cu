#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

__global__ void VecAdd(float* A, float* B, float*
    C, int N_op,int op_loop){

    // N_op : no of total ops
    // op_loop: no of ops to do in a loop
    // Host code
    int j;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N_op){
        for (j=0;j<op_loop;j++){
            C[i*op_loop+j] = A[i*op_loop+j] + B[i*op_loop+j];
        }
    }

    }

int main(int argc,char **argv) {
    //printf("Usage example. \n./ee16b068_456.out <threads_per_block> <ops_per_thread> <vector_size> <filename1> <filename2>");

    if (argc<5){
        //printf("Too few arguments.\nUsage is ./ee16b068_456.out <threads_per_block> <ops_per_thread> <vector_size> <filename1> <filename2> ");
        return 1;
    }
    
    int threadsPerBlock_op=atoi(argv[1]);
    int op_loop=atoi(argv[2]);
    //int N = pow(2,atoi(argv[3]));
    int N = atoi(argv[3]);

    FILE *arrayfile_a;
    FILE *arrayfile_b;

    size_t size = N * sizeof(float);

    //Helper variables
    int loop;
    float time_spent;

    //files
    arrayfile_a = fopen(argv[4], "r");
    arrayfile_b = fopen(argv[5], "r");

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Read first two arrays
    //printf("\nArray A (first 10 values) \n ");
    for (loop = 0; loop < N; loop++)
    {
        fscanf(arrayfile_a, "%f", &h_A[loop]);
        //if (loop<10){
         //   printf("%f ", h_A[loop]);
        //}
    }

    //printf("\nArray B (first 10 values) \n ");
    for (loop = 0; loop < N; loop++)
    {
        fscanf(arrayfile_b, "%f", &h_B[loop]);
        //if (loop<10){
            //printf("%f ", h_B[loop]);
        //}
    }

    // Allocate vectors in device memory
    float* d_A;   cudaMalloc(&d_A, size);
    float* d_B;   cudaMalloc(&d_B, size);
    float* d_C;   cudaMalloc(&d_C, size);

    //GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size,cudaMemcpyHostToDevice);


    //ops per loop
    //printf("Ops per loop %d\n",op_loop);
    //printf("Threads per Block %d\n",threadsPerBlock_op);
    //printf("Vector Size %d\n",op_loop);

    // Invoke kernel
    int threadsPerBlock = threadsPerBlock_op;
    int N_op=(N + op_loop -1)/op_loop;
    int blocksPerGrid = (N_op + threadsPerBlock - 1) /threadsPerBlock;

    cudaEventRecord(start, 0);
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B, d_C, N_op,op_loop);
    cudaEventRecord(stop, 0);

    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size,cudaMemcpyDeviceToHost);

    //printf("\nArray C (first 10 outputs)\n");
    //for(loop = 0; loop < 10; loop++)
    //    printf("%f ", h_C[loop]);

    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_spent, start, stop);
    printf("%f",time_spent);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // Free host memory
    free(h_A);
    free(h_B);
    return 0;
}