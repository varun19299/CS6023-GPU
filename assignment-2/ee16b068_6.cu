#include <stdlib.h>
#include <stdio.h>
#define TILE_WIDTH (16)

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
  const char *fname = "assignment2_6_out";
  FILE *f = fopen(fname, "w");
  for(unsigned i=0; i < numRows; i++)
    {
        for(unsigned j=0; j < numCols; j++)
        fprintf(f,"%4.4f ", mat[i*numCols + j]);
        fprintf(f,"\n");
    }
    fclose(f); }

//template<int TILE_WIDTH>
__global__ void MatrixMulKernel_col_maj(double* M, double* N, double* Q, int M_r, int N_c, int M_c) { 
    //extern __shared__ double buffer[];
    //double *ds_M = &buffer[0];
    //double *ds_N = &buffer[Width*Width];

    __shared__ double ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ double ds_N[TILE_WIDTH][TILE_WIDTH];

    // Generate IDs
    double Pvalue=0;
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    
    // Loop over the M and N tiles required to compute the P element
    for (int p = 0; p < (M_c)/TILE_WIDTH; ++p) {
        if ( (Row < M_r) && (tx + p*TILE_WIDTH) < M_c){
        // Collaborative loading of M and N tiles into shared memory
        ds_M[ty][tx] = M[Row*M_c + p*TILE_WIDTH+tx];
        }
        else{
            ds_M[ty][tx]=0.0;
        }
        if ( (Col < N_c) && (ty + p*TILE_WIDTH) < M_c){
        ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*N_c + Col];
        }
        else{
            ds_N[ty][tx]=0.0;
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i){
            Pvalue += ds_M[ty][i] * ds_N[i][tx];
            
        }
        __syncthreads();
        
    }
    
    if ((Row < M_r) && (Col < N_c)){
    Q[Row*N_c+Col] = Pvalue;
    }
}

int main(int argc,char **argv) {
    int M_r,M_c,N_c;

    M_r=4096;
    M_c=8192;
    N_c=16384;    // M_c=N_r

    int loop1, loop2; // loop variables
    float time_spent;

    size_t size1 = M_r *M_c* sizeof(double);
    size_t size2 = M_c *N_c* sizeof(double);
    size_t size3 = M_r *N_c* sizeof(double);

    double*h_matA = (double*)malloc(size1);
    double*h_matB = (double*)malloc(size2);
    double*h_matC = (double*)malloc(size3); // result

    fill_matrix(h_matA,M_r,M_c);
    fill_matrix(h_matB,M_c,N_c);

    printf("\nMatrix A (first 10*10 inputs)\n");
    for(loop1 = 0; loop1 < 10; loop1++){
        for (loop2=0;loop2 < 10; loop2++)
            printf("%f ", *(h_matA + M_c*loop1 + loop2));
        printf("\n");
    }

    printf("\n\nMatrix B (first 10*10 inputs)\n");
    for(loop1 = 0; loop1 < 10; loop1++){
        for (loop2=0;loop2 < 10; loop2++)
            printf("%f ", *(h_matB + N_c*loop1 + loop2));
        printf("\n");
    }

    double* d_matA;   cudaMalloc(&d_matA, size1);
    double* d_matB;   cudaMalloc(&d_matB, size2);
    double* d_matC;   cudaMalloc(&d_matC, size3);

    //GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_matA, h_matA, size1,cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, size2,cudaMemcpyHostToDevice);

    // Invoke kernel
    dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
    dim3 blocksPerGrid ((M_r + threadsPerBlock.x-1) /threadsPerBlock.x,(N_c+ threadsPerBlock.y-1) /threadsPerBlock.y);

    cudaEventRecord(start, 0);
    MatrixMulKernel_col_maj<<<blocksPerGrid, threadsPerBlock>>>(d_matA,d_matB, d_matC, M_r,N_c,M_c);
    //cudaDeviceSynchronize();//To synchronize the device
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_spent, start, stop);
    printf("\nTime spent in col maj %f\n",time_spent);

    // h_C contains the result in host memory
    cudaMemcpy(h_matC, d_matC, size3,cudaMemcpyDeviceToHost);

    printf("\n\nMatrix C (first 10*10 outputs)\n");
    for(loop1 = 0; loop1 < 10; loop1++){
        for (loop2=0;loop2 < 10; loop2++)
            printf("%f ", *(h_matC + N_c*loop1 + loop2));
        printf("\n");
    }

    // Log outputs
    printf("\nWritting to file assignment_2_1_out as Mat C");
    print_matrix_to_file(h_matC,M_r,N_c);

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