#include <stdlib.h>
#include <stdio.h>

void fill_matrix(double *mat, unsigned numRows, unsigned numCols)
{
    for(unsigned i=0; i < numRows; i++)
       for(unsigned j=0; j < numCols; j++)
       {
         mat[i*numCols + j] = i*2.1f + j*3.2f;
       }
}

void print_matrix_to_file(double *mat, unsigned numRows, unsigned numCols)
{
  const char *fname = "assignment2_1_out";
  FILE *f = fopen(fname, "w");
  for(unsigned i=0; i < numRows; i++)
  {
     for(unsigned j=0; j < numCols; j++)
     fprintf(f,"%4.4f ", mat[i*numCols + j]);
     fprintf(f,"\n");
}
fclose(f); }

__global__ void MatrixMulKernel_col_maj(double* M, double* N, double* P, int Width) { 
    // Calculate the row index of the P element and M
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    // Calculate the column index of P and N
    int Col = blockIdx.x*blockDim.x+threadIdx.x; 
    
    if ((Row < Width) && (Col < Width)) {
            float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[Row*Width+k]*N[k*Width+Col];
        }
            P[Row*Width+Col] = Pvalue;
        }
    }

__global__ void MatrixMulKernel_row_maj(double* M, double* N, double* P, int Width) { 
    // Calculate the row index of the P element and M
    int Row = blockIdx.y*blockDim.y+threadIdx.x;
    // Calculate the column index of P and N
    int Col = blockIdx.x*blockDim.x+threadIdx.y; 
    
    if ((Row < Width) && (Col < Width)) {
            float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[Row*Width+k]*N[k*Width+Col];
        }
            P[Row*Width+Col] = Pvalue;
        }
    }

int main(int argc,char **argv) {
    int N = 8192;
    size_t size = N *N* sizeof(double);

    int thread_dim_ll[8];
    int thread_dim;

    double*h_matA = (double*)malloc(size);
    double*h_matB = (double*)malloc(size);
    double*h_matC = (double*)malloc(size); // result

    int loop, loop1, loop2; // loop variables
    float time_spent;

    fill_matrix(h_matA,N,N);
    fill_matrix(h_matB,N,N);

    printf("Thread dims\n");
    for (loop=0;loop<8;loop++){
        thread_dim_ll[loop]=pow(2,2+loop);
    }

    printf("\nMatrix A (first 10*10 inputs)\n");
    for(loop1 = 0; loop1 < 10; loop1++){
        for (loop2=0;loop2 < 10; loop2++)
            printf("%f ", *(h_matA + N*loop1 + loop2));
    }

    printf("\n\nMatrix B (first 10*10 inputs)\n");
    for(loop1 = 0; loop1 < 10; loop1++){
        for (loop2=0;loop2 < 10; loop2++)
            printf("%f ", *(h_matB + N*loop1 + loop2));
    }

    double* d_matA;   cudaMalloc(&d_matA, size);
    double* d_matB;   cudaMalloc(&d_matB, size);
    double* d_matC;   cudaMalloc(&d_matC, size);

    //GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_matA, h_matA, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, size,cudaMemcpyHostToDevice);

    for (loop = 0; loop < 8; loop++){

        //thread dim
        thread_dim=thread_dim_ll[loop];
        // Invoke kernel
        dim3 threadsPerBlock = (thread_dim,thread_dim);
        dim3 blocksPerGrid ((N + threadsPerBlock.x - 1) /threadsPerBlock.x,(N + threadsPerBlock.y - 1) /threadsPerBlock.y);

        cudaEventRecord(start, 0);
        MatrixMulKernel_col_maj<<<blocksPerGrid, threadsPerBlock>>>(d_matA,d_matB, d_matC, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_spent, start, stop);
        printf("\nTime spent in col maj %f with threadsPerBlock %d \n",time_spent,thread_dim);

    }
    // h_C contains the result in host memory
    cudaMemcpy(h_matC, d_matC, size,cudaMemcpyDeviceToHost);

    printf("\n\nMatrix C (first 10*10 outputs)\n");
    for(loop1 = 0; loop1 < 10; loop1++){
        for (loop2=0;loop2 < 10; loop2++)
            printf("%f ", *(h_matC + N*loop1 + loop2));
    }
      

    // Log outputs
    printf("\nWritting to file assignment_2_1_out as Mat C");
    print_matrix_to_file(h_matC,N,N);

    // Free device memory
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);

    // Free host memory
    free(h_matA);
    free(h_matB);
    free(h_matC);
    return 0;
}