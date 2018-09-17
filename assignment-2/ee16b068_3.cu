#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>


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
  const char *fname = "assignment2_3_out";
  FILE *f = fopen(fname, "w");
  for(unsigned i=0; i < numRows; i++)
  {
     for(unsigned j=0; j < numCols; j++)
     fprintf(f,"%4.4f ", mat[i*numCols + j]);
     fprintf(f,"\n");
}
fclose(f); }

template<int TILE_WIDTH>
__global__ void MatrixMulKernel_col_maj(double* M, double* N, double* P, int Width) { 
    extern __shared__ double buffer[];

    double *ds_M = &buffer[0]; // TILE_WIDTH WIDTH
    double *ds_N = &buffer[TILE_WIDTH*Width]; // WIDTH TILE_WIDTH

    //__shared__ float ds_M[Width][Width];
    //__shared__ float ds_N[Width][Width];
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    
    
    // Loop over the M and N tiles required to compute the P element
    for (int p = 0; p < Width/TILE_WIDTH; ++p) {
    // Collaborative loading of M and N tiles into shared memory
    ds_M[ty*Width + tx + p*blockDim.x ] = M[Row*Width + p*TILE_WIDTH+tx];
    ds_N[ty*TILE_WIDTH + blockDim.y*TILE_WIDTH*p + tx] = N[(p*TILE_WIDTH+ty)*Width + Col];
    __syncthreads();
    }

    double Pvalue = 0;
    for (int i = 0; i < TILE_WIDTH; ++i){
        Pvalue += ds_M[ty*Width + i] * ds_N[i*Width + tx];
    }
    __syncthreads();
    P[Row*Width+Col] = Pvalue;
}

int main(int argc,char **argv) {
    int N_ll[2]; int N;
    

    int loop, loop1, loop2; // loop variables
    float time_spent;

    N_ll[0]=16; N_ll[1]=8192;

    for (loop=0;loop<2;loop++){

        N=N_ll[loop];

        size_t size = N *N* sizeof(double);

        double*h_matA = (double*)malloc(size);
        double*h_matB = (double*)malloc(size);
        double*h_matC = (double*)malloc(size); // result

        fill_matrix(h_matA,N,N);
        fill_matrix(h_matB,N,N);

        printf("\nMatrix A (first 10*10 inputs)\n");
        for(loop1 = 0; loop1 < 10; loop1++){
            for (loop2=0;loop2 < 10; loop2++)
                printf("%f ", *(h_matA + N*loop1 + loop2));
            printf("\n");
        }

        printf("\n\nMatrix B (first 10*10 inputs)\n");
        for(loop1 = 0; loop1 < 10; loop1++){
            for (loop2=0;loop2 < 10; loop2++)
                printf("%f ", *(h_matB + N*loop1 + loop2));
            printf("\n");
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

        // Invoke kernel
        dim3 threadsPerBlock (16,16);
        dim3 blocksPerGrid ((N + threadsPerBlock.x - 1) /threadsPerBlock.x,(N + threadsPerBlock.y - 1) /threadsPerBlock.y);

        cudaEventRecord(start, 0);
        size_t blocksize = 2 * N * 16;
        (MatrixMulKernel_col_maj<16>)<<<blocksPerGrid, threadsPerBlock, sizeof(double)*blocksize>>>(d_matA,d_matB, d_matC, N);
        cudaError_t err1 = cudaPeekAtLastError();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_spent, start, stop);
        printf("\nTime spent in col maj %f\n",time_spent);

        // h_C contains the result in host memory
        cudaMemcpy(h_matC, d_matC, size,cudaMemcpyDeviceToHost);

        printf("\n\nMatrix C (first 10*10 outputs)\n");
        for(loop1 = 0; loop1 < 10; loop1++){
            for (loop2=0;loop2 < 10; loop2++)
                printf("%f ", *(h_matC + N*loop1 + loop2));
            printf("\n");
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
        
        
    }   
    return 0;
}
