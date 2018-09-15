#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

__global__ void VecAdd(float* A, float* B, float*
    C, int N){
    // Host code
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];

    }

int main() {
    int N = pow(2,15);
    int avg_loop=1000;
    
    int threadsPerBlock_array[6];

    size_t size = N * sizeof(float);

    //Helper variables
    int loop;
    int thread_loop;
    int clock_loop;
    float time_spent;

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    printf("\nThreads per Block per array\n");
    for(loop = 5; loop < 11; loop++){
        threadsPerBlock_array[loop-5] = pow(2,loop);
        printf("%d ", threadsPerBlock_array[loop-5]);
    }

    // Initialize input vectors
    printf("Array A (first 10 values) \n ");
    for(loop = 0; loop < N; loop++){
    h_A[loop] = rand() % 100 + 1;
    if (loop<10){
        printf("%f ", h_A[loop]);
    }
    }

    printf("\nArray B (first 10 values) \n ");
    for(loop = 0; loop < N; loop++){
        

        h_B[loop] = rand() % 100 + 1;
        if (loop<10){
            printf("%f ", h_B[loop]);
        }
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

        for (thread_loop=0;thread_loop<6;thread_loop++){

            
            for(clock_loop=0;clock_loop<avg_loop;clock_loop++){

                if (clock_loop==1){
                    cudaEventRecord(start, 0);
                }
            
            // Invoke kernel
            int threadsPerBlock = threadsPerBlock_array[thread_loop];
            int blocksPerGrid = (N + threadsPerBlock - 1) /threadsPerBlock;

            
            VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B, d_C, N);

            // h_C contains the result in host memory
            cudaMemcpy(h_C, d_C, size,cudaMemcpyDeviceToHost);

            //printf("\nArray C (first 10 outputs)\n");
            //for(loop = 0; loop < 10; loop++)
              //  printf("%f ", h_C[loop]);
            }

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_spent, start, stop);
            time_spent=time_spent/(avg_loop-1)*10;

            printf("\n Average Time spent in loop %d is %f",thread_loop,time_spent);
        }
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // Free host memory
    free(h_A);
    free(h_B);
    return 0;
}