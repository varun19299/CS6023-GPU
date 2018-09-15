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

int main() {
    int N_array[6];
    
    int threadsPerBlock_op=256;
    int op_loop=4;

    int loop;
    int N_loop;
    int avg_loop;
    float time_spent;
    int clock_loop;

    avg_loop=100;
    //GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (N_loop=0;N_loop<6;N_loop++){
        // Size of vector array
        N_array[N_loop]=pow(2,15+N_loop);
    }

    for (N_loop=0;N_loop<6;N_loop++){

        

            
        
            // size of array
            size_t size = N_array[N_loop]* sizeof(float);

            // Allocate input vectors h_A and h_B in host memory
            float* h_A = (float*)malloc(size);
            float* h_B = (float*)malloc(size);
            float* h_C = (float*)malloc(size);

            // Initialize input vectors
        // printf("Array A (first 10 values) \n ");
            for(loop = 0; loop < N_array[N_loop]; loop++){
            h_A[loop] = rand() % 100 + 1;
            //if (loop<10){
        //     printf("%f ", h_A[loop]);
        // }
            }

            //printf("\nArray B (first 10 values) \n ");
            for(loop = 0; loop < N_array[N_loop]; loop++){
                h_B[loop] = rand() % 100 + 1;
                //if (loop<10){
            //       printf("%f ", h_B[loop]);
            //}
            }

            // Allocate vectors in device memory
            float* d_A;   cudaMalloc(&d_A, size);
            float* d_B;   cudaMalloc(&d_B, size);
            float* d_C;   cudaMalloc(&d_C, size);

            

            // Copy vectors from host memory to device memory
            cudaMemcpy(d_A, h_A, size,cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, size,cudaMemcpyHostToDevice);       

            //vector size per loop
            //printf("Vector size per loop %d",N_array[N_loop]);

            // Invoke kernel
            int threadsPerBlock = threadsPerBlock_op;
            int N_op=(N_array[N_loop] + op_loop -1)/op_loop;
            int blocksPerGrid = (N_op + threadsPerBlock - 1) /threadsPerBlock;

            for(clock_loop=0;clock_loop<avg_loop;clock_loop++){

                if (clock_loop==1){
                    cudaEventRecord(start, 0);
                }

                VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B, d_C, N_op,op_loop);

                if (clock_loop==avg_loop-1){
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&time_spent, start, stop);
                }    

            }

        // h_C contains the result in host memory
        cudaMemcpy(h_C, d_C, size,cudaMemcpyDeviceToHost);

        //printf("\nArray C (first 10 outputs)\n");
        //for(loop = 0; loop < 10; loop++)
        //   printf("%f ", h_C[loop]);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        // Free host memory
        free(h_A);
        free(h_B);

    
        
        time_spent=time_spent/(avg_loop-1)*10;

        printf("\n Average Time spent in loop %d is %f",N_loop,time_spent);

        }
    return 0;
}