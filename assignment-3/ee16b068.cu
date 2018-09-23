#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

#define MAXWORDS 20000
//#define MAXWORDS 10

void print_to_file(int *h_hist, int N)
{
  const char *fname = "assignment3_out";
  FILE *f = fopen(fname, "w");
  for(unsigned i=0; i < numRows; i++)
  {
     for(unsigned j=0; j < numCols; j++)
     fprintf(f,"%4.4f ", mat[i*numCols + j]);
     fprintf(f,"\n");
}
int loop, loop1;
    for(loop = 0; loop < pow(20,N); loop++){
        if (h_hist[loop]>0){
            a=loop/(int)pow(20,N-1);
            b=loop;
            fprintf(f,"%d ",a+1);
            for (loop1=1;loop1 < N; loop1++){
                a=b-a*(int)pow(20,N-loop1);
                b=a;
                a/=(int)pow(20,N-loop1-1);
                fprintf(f,"%d  ", a+1);
            }
            fprintf(f,"%d \n",h_hist[loop]);
        }
    }
fclose(f); }

int checkWord(char* word,char* words,int* count_array,int offset){
    // Check if word meets, else pre-process
    // Args:
    // word >> word of consideration from fscanf
    // words >> Array where, every 20 chars is a word
    // offset >> Which entry to start writting at (modulo 20)
    // Returns:
    // new offset
    // Modifies:
    // words
    int loop=0;
    int count=0;

    for (loop=0;loop<strlen(word)-1;loop++)
    {
       
       if (word[loop]=='-')
       {
          words[offset*20+loop]=0;
          printf("Word %s \n",&words[offset*20]);
          offset+=1;
          count_array[offset]=count;
          count=0;
        }
       else{
          /* Copy character */
          words[offset*20+loop]=word[loop];
          count+=1;
       }
    }

    if (ispunct((unsigned char)word[strlen(word)-1]))
       {
          /* Skip this character */
          words[offset*20+strlen(word)-1]=0;
          count_array[offset]=count;
          offset+=1;
       }
    else{
        words[offset*20+strlen(word)-1]=word[strlen(word)-1];
        count+=1;
        words[offset*20+strlen(word)]=0;
        count_array[offset]=count;
        offset+=1;
    }
    return offset;

}

__global__ void nCountGram_optimal(int* d_count, int* d_hist, int N, int totalWordCount, int sub_hist_size){
    extern __shared__ int buffer[];
    int *temp = &buffer[0];

    //__shared__ int temp[1024];
    // Helper var
    int index, j, p;
    int a, b;

    a=1;
    for (p=0;p<N;p++){
        a*=20;
    }

    for (p=0; p<sub_hist_size/1024 +1; p++){
        if (threadIdx.x + p*1024 < sub_hist_size){
            temp[threadIdx.x + p*1024] = 0;
        }
    }

    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x ;//blockIdx.y*gridDim.y;
    int offset = blockDim.x * gridDim.x*blockIdx.y*gridDim.y;

    while (i < totalWordCount - N + 1)
    {
        // Since 0,0 is invalid
        index=-1;
        b=a/20;
        for (j = 0;j < N; j++){
            index+=(d_count[i+j])*b;
            b/=20;
        }
        if ((index<sub_hist_size*(blockIdx.y+1)) && (index > sub_hist_size*blockIdx.y)){
            //printf("Index %d",index);
        atomicAdd( &temp[index - blockIdx.y*sub_hist_size], 1);
        }
        i += offset;
    }

    __syncthreads();

    for (p=0;p<sub_hist_size/1024+1;p++){
        if (threadIdx.x + p*1024 < sub_hist_size){
            atomicAdd( &(d_hist[threadIdx.x + sub_hist_size*blockIdx.y + p*1024]), temp[threadIdx.x + p*1024] );
            if (d_hist[threadIdx.x+ sub_hist_size*blockIdx.y + p*1024]>0){
                printf("Hist val at %d is %d \n",threadIdx.x+sub_hist_size*blockIdx.y+p*1024,d_hist[threadIdx.x +sub_hist_size*blockIdx.y+ p*1024]);
            }
        }
    }

    __syncthreads();

}

__global__ void nCountGram(int* d_count, int* d_hist, int N, int totalWordCount){
    extern __shared__ int buffer[];
    int *temp = &buffer[0];

    //__shared__ int temp[1024];
    // Helper var
    int index, j, p;
    int a, b;

    a=1;
    for (p=0;p<N;p++){
        a*=20;
    }
    printf("t %d",threadIdx.x + 1*1024);


    for (p=0; p<a/1024+1; p++){
        if (threadIdx.x + p*1024< a){
            temp[threadIdx.x + p*1024] = 0;
        }
    }

    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    printf("Offset %d",offset);
    while (i < totalWordCount - N + 1)
    {
        // Since 0,0 is invalid
        index=-1;
        b=a/20;
        for (j = 0;j < N; j++){
            index+=(d_count[i+j])*b;
            b/=20;
        }
        atomicAdd( &temp[index], 1);
        i += offset;
        printf("Index %d",index);
    }

    __syncthreads();

    for (p=0;p<a/1024+1;p++){
        if (threadIdx.x + p*1024< a){
            atomicAdd( &(d_hist[threadIdx.x + p*1024]), temp[threadIdx.x + p*1024] );
            if (temp[threadIdx.x+p*1024]>0){
                //printf("Hist val at %d is %d \n",threadIdx.x+p*1024,d_hist[threadIdx.x + p*1024]);
            }
        }
    }

    __syncthreads();

}

int main(int argc,char **argv) {
    // Helper vars
    int loop, loop1, a, b;
    int sub_hists, sub_hist_size;
    float time_spent;

    int N = atoi(argv[1]);
    char *filename = argv[2];
    int count_array[MAXWORDS];
    char words[MAXWORDS * 20];

    // For calculating N-count-grams
    // Filename: shaks.txt
    // Stores all words in 1D array
    // Single word length is bounded by 20
    // Take input string into this
    char curWord[20];
    int totalWordCount = 0;
    FILE *ipf;
    ipf = fopen(filename, "r");
    while (fscanf(ipf, "%s ", curWord) != EOF && totalWordCount < MAXWORDS) {
         // Count of number of words read
        //printf("Curr word %s \n",curWord);
        totalWordCount=checkWord(curWord,words,count_array,totalWordCount);
        //printf("Current word %s \n",curWord);
        //printf("Word count %d \n",totalWordCount);
    }
    fclose(ipf);

    //for (loop=0;loop<totalWordCount;loop++){
    //    printf("Word %d %s ",loop,&words[20*loop]);
    //    printf("Char count %d \n",count_array[loop]);
    //}
    // Check for word properties
    // and update ‘words[]’ array.
    // Modify this section according
    // to below mentioned properties

    //Create CPU arrays (hist)
    int* h_hist = (int*)malloc((int)pow(20,N)*sizeof(int));

    // Create GPU arrays, Copy count array from host memory to device memory
    int* d_count; cudaMalloc(&d_count, MAXWORDS*sizeof(int));
    cudaMemcpy(d_count, count_array, MAXWORDS*sizeof(int),cudaMemcpyHostToDevice);
    
    int* d_hist; cudaMalloc(&d_hist, (int)pow(20,N)*sizeof(int));

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // // Invoke kernel
    if (N<5){
        int threadsPerBlock = 1024;
        int blocksPerGrid = ((pow(20,N) + threadsPerBlock - 1) /threadsPerBlock);
        cudaEventRecord(start, 0);
        nCountGram<<<blocksPerGrid, threadsPerBlock, (unsigned int)pow(20,N)*sizeof(int)>>>(d_count,d_hist, N,totalWordCount);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_spent, start, stop);
        printf("\nTime spent in hist binning %f\n",time_spent);
    }
    else{
        sub_hist_size=160000;
        sub_hists=(int)pow(20,N)/sub_hist_size;
        int threadsPerBlock = 1024;
        dim3 blocksPerGrid = (((pow(20,N)/sub_hists + threadsPerBlock - 1) /threadsPerBlock),sub_hists);
        cudaEventRecord(start, 0);
        nCountGram_optimal<<<blocksPerGrid, threadsPerBlock, sub_hist_size*sizeof(int)>>>(d_count,d_hist, N,totalWordCount,sub_hist_size);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_spent, start, stop);
        printf("\nTime spent in hist binning %f\n",time_spent);
    }
    

    // h_hist contains the result in host memory
    cudaMemcpy(h_hist, d_hist, (int)pow(20,N)*sizeof(int),cudaMemcpyDeviceToHost);
    print_to_file(h_hist,N);

    // printf("\n\n Histogram for N of value %d, total number of words %d \n",N,totalWordCount);
    // for(loop = 0; loop < pow(20,N); loop++){
    //     if (h_hist[loop]>0){
    //         a=loop/(int)pow(20,N-1);
    //         b=loop;
    //         printf("Value %d ",a+1);
    //         for (loop1=1;loop1 < N; loop1++){
    //             a=b-a*(int)pow(20,N-loop1);
    //             b=a;
    //             a/=(int)pow(20,N-loop1-1);
    //             printf("%d  ", a+1);
    //         }
    //         printf(" Count: %d \n",h_hist[loop]);
    //     }
    // }

    // Free device memory
    cudaFree(d_count);
    cudaFree(d_hist);

    // Free host memory
    free(h_hist);
    return 0;
}
