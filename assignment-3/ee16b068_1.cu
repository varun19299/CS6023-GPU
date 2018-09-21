#include <stdlib.h>
#include <stdio.h>
#define MAXWORDS 20000

bool checkWord(word){
    // Check if word meets, else pre-process
}

__global__ void windowGram(char** word){

}

int main(int argc,char **argv) {
    
    //size_t size = N *N* sizeof(double);

    int N = atoi(argv[1]);
    char *filename = argv[2];
    char words[MAXWORDS * 20];
    // For calculating N-count-grams
    // Filename: shaks.txt
    // Stores all words in 1D array
    // Single word length is bounded by 20
    // Take input string into this
    char curWord[40];
    int totalWordCount = 0;
    FILE *ipf = fopen(filename, “r”);
    while (fscanf(ipf, “%s ”, curWord) != EOF
                    // Count of number of words read
    && totalWordCount < MAXWORDS) {
        //checkWord(curWord, ...);
        printf("Current word %s \n",curWord);
        totalWordCount += 1;
    }
    fclose(ipf);
    // Check for word properties
    // and update ‘words[]’ array.
    // Modify this section according
    // to below mentioned properties

    // cudaMalloc(&d_matC, size);

    // //GPU timing
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // // Copy vectors from host memory to device memory
    // cudaMemcpy(d_matA, h_matA, size,cudaMemcpyHostToDevice);
    // cudaMemcpy(d_matB, h_matB, size,cudaMemcpyHostToDevice);

    // // Invoke kernel
    // dim3 threadsPerBlock = (16,16);
    // dim3 blocksPerGrid ((N + threadsPerBlock.x - 1) /threadsPerBlock.x,(N + threadsPerBlock.y - 1) /threadsPerBlock.y);

    // cudaEventRecord(start, 0);
    // MatrixMulKernel_col_maj<<<blocksPerGrid, threadsPerBlock>>>(d_matA,d_matB, d_matC, N);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time_spent, start, stop);
    // printf("\nTime spent in col maj %f\n",time_spent);

    // // h_C contains the result in host memory
    // cudaMemcpy(h_matC, d_matC, size,cudaMemcpyDeviceToHost);

    // printf("\n\nMatrix C via col major (first 10*10 outputs)\n");
    // for(loop1 = 0; loop1 < 10; loop1++){
    //     for (loop2=0;loop2 < 10; loop2++)
    //         printf("%f ", *(h_matC + N*loop1 + loop2));
    // }

    // cudaEventRecord(start, 0);
    // MatrixMulKernel_row_maj<<<blocksPerGrid, threadsPerBlock>>>(d_matA,d_matB, d_matC, N);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time_spent, start, stop);
    // printf("\nTime spent in row maj %f\n",time_spent);

    // // h_C contains the result in host memory
    // cudaMemcpy(h_matC, d_matC, size,cudaMemcpyDeviceToHost);

    // printf("\n\nMatrix C via row major (first 10*10 outputs)\n");
    // for(loop1 = 0; loop1 < 10; loop1++){
    //     for (loop2=0;loop2 < 10; loop2++)
    //         printf("%f ", *(h_matC + N*loop1 + loop2));
    // }
      

    // // Log outputs
    // printf("\nWritting to file assignment_2_1_out as Mat C");
    // print_matrix_to_file(h_matC,N,N);

    // // Free device memory
    // cudaFree(d_matA);
    // cudaFree(d_matB);
    // cudaFree(d_matC);

    // // Free host memory
    // free(h_matA);
    // free(h_matB);
    // free(h_matC);
    return 0;
}
