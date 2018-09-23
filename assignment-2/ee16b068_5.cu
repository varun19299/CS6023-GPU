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
  const char *fname = "assignment2_5_out";
  FILE *f = fopen(fname, "w");
  for(unsigned i=0; i < numRows; i++)
    {
        for(unsigned j=0; j < numCols; j++)
        fprintf(f,"%4.4f ", mat[i*numCols + j]);
        fprintf(f,"\n");
    }
    fclose(f); }

__global__ void MatrixMulKernel_col_maj(double* M, double* N, double* P, int Width, int TILE_WIDTH) { 
    extern __shared__ double buffer[];
    double *ds_M = &buffer[0];
    double *ds_N = &buffer[TILE_WIDTH*TILE_WIDTH];

    //__shared__ double ds_M[TILE_WIDTH][TILE_WIDTH];
    //__shared__ double ds_N[TILE_WIDTH][TILE_WIDTH];

    // Generate IDs
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    ds_M[ty*TILE_WIDTH + tx] = 0.0;
    ds_N[ty*TILE_WIDTH + tx] = 0.0;
    
    double Pvalue=0;
    // Loop over the M and N tiles required to compute the P element
    for (int p = 0; p < (Width-1)/TILE_WIDTH+1; ++p) {
        if ( (Row < Width) && (threadIdx.x + (p*TILE_WIDTH)) < Width){
        // Collaborative loading of M and N tiles into shared memory
        ds_M[ty*TILE_WIDTH + tx] = M[Row*Width + p*TILE_WIDTH+tx];
        ds_N[ty*TILE_WIDTH + tx] = N[(p*TILE_WIDTH+ty)*Width + Col];
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
        Pvalue += ds_M[ty*TILE_WIDTH + i] * ds_N[i*TILE_WIDTH + tx];
        __syncthreads();
    }

    if (Row < Width && Col < Width){
    P[Row*Width+Col] = Pvalue;
    }
}

int main(int argc,char **argv) {
    int N;
    int TILE_WIDTH_ll[4], TILE_WIDTH;
    
    int loop,loop1, loop2; // loop variables
    float time_spent;

    N=8192;

    size_t size = N *N* sizeof(double);

    double*h_matA = (double*)malloc(size);
    double*h_matB = (double*)malloc(size);
    double*h_matC = (double*)malloc(size); // result

    fill_matrix(h_matA,N,N);
    fill_matrix(h_matB,N,N);

    for (loop = 0; loop<4; loop++){
        TILE_WIDTH_ll[loop]=pow(2,2+loop);
    }

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

    for (loop =0;loop < 4; loop++){

        TILE_WIDTH=TILE_WIDTH_ll[loop];
        
        // Invoke kernel
        dim3 threadsPerBlock (TILE_WIDTH,TILE_WIDTH,1);
        dim3 blocksPerGrid ((N + threadsPerBlock.x) /threadsPerBlock.x,(N + threadsPerBlock.y) /threadsPerBlock.y,1);

        size_t blocksize = 2 * TILE_WIDTH * TILE_WIDTH;

        cudaEventRecord(start, 0); 
        MatrixMulKernel_col_maj<<<blocksPerGrid, threadsPerBlock, sizeof(double)*blocksize>>>(d_matA,d_matB, d_matC, N, TILE_WIDTH);
        //cudaDeviceSynchronize();//To synchronize the device
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_spent, start, stop);
        printf("\nTime spent in col maj for tile %d x %d %f\n",TILE_WIDTH,TILE_WIDTH,time_spent);
        
        // h_C contains the result in host memory
        cudaMemcpy(h_matC, d_matC, size,cudaMemcpyDeviceToHost);

        printf("\n\nMatrix C (first 10*10 outputs)\n");
        for(loop1 = 0; loop1 < 10; loop1++){
            for (loop2=0;loop2 < 10; loop2++)
                printf("%f ", *(h_matC + N*loop1 + loop2));
            printf("\n");
        }
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