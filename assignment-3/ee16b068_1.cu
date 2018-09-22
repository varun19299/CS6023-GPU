#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

#define MAXWORDS 20000

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

__global__ void nCountGram(int* d_count, int* d_hist, int N){
    extern __shared__ unsigned int buffer[];
    unsigned int *temp = &buffer[0];
    //__shared__ unsigned int temp[1024];

    // Helper var
    int index, j, p;

    int a;
    for (p=0;p<N;p++){
        a*=20;
    }

    int b;

    for (p=0;p<a/1024+1;p++){
    temp[threadIdx.x + p*1024] = 0;
    }
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while (i < a)
    {
        // Since 0,0 is invalid
        index=-1;
        b=a/20;
        for (j = 0;j < N; j++){
            index+=d_count[i+j]*b;
            b/=20;
        }
    atomicAdd( &temp[index], 1);
    i += offset;
    }

    __syncthreads();

    for (p=0;p<a/1024+1;p++){
        atomicAdd( &(d_hist[threadIdx.x + p*1024]), temp[threadIdx.x + p*1024] );
        }
}

int main(int argc,char **argv) {
    // Helper vars
    int loop, loop1;
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
        printf("Curr word %s \n",curWord);
        totalWordCount=checkWord(curWord,words,count_array,totalWordCount);
        //printf("Current word %s \n",curWord);
        //printf("Word count %d \n",totalWordCount);
    }
    fclose(ipf);

    for (loop=0;loop<totalWordCount;loop++){
        printf("Word %d %s \n",loop,&words[20*loop]);
        printf("Char count %d \n",count_array[loop]);
    }
    // Check for word properties
    // and update ‘words[]’ array.
    // Modify this section according
    // to below mentioned properties

    //Create CPU arrays (hist)
    unsigned int* h_hist = (unsigned int*)malloc(pow(20,N)*sizeof(unsigned int));

    // Create GPU arrays, Copy count array from host memory to device memory
    int* d_count; cudaMalloc(&d_count, MAXWORDS*sizeof(int));
    cudaMemcpy(d_count, count_array, MAXWORDS,cudaMemcpyHostToDevice);
    int* d_hist; cudaMalloc(&d_hist, pow(20,N)*sizeof(unsigned int));

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // // Invoke kernel
    dim3 threadsPerBlock = 1024;
    dim3 blocksPerGrid ((pow(20,N) + threadsPerBlock.x - 1) /threadsPerBlock.x);

    cudaEventRecord(start, 0);
    nCountGram<<<blocksPerGrid, threadsPerBlock, pow(N,20)*sizeof(unsigned int)>>>(d_count,d_hist, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_spent, start, stop);
    printf("\nTime spent in col maj %f\n",time_spent);

    // h_hist contains the result in host memory
    cudaMemcpy(h_hist, d_hist, pow(20,N)*sizeof(unsigned int),cudaMemcpyDeviceToHost);

    printf("\n\n Histogram for N of value %d, total number of words %d \n",N,totalWordCount);
    for(loop = 0; loop < pow(20,N); loop++){
        if (h_hist[loop]>0){
        printf("Value ");
        a=loop/(unsigned int)pow(20,N-1);
         for (loop1=1;loop1 < N; loop1++)
            printf("%u  ", a);
            a-=a*(unsigned int)pow(20,N-loop1);
            a/=(unsigned int)pow(20,N-loop1-1)

        printf(" Count: %u \n",h_hist[loop]);
        }
     }

    // Free device memory
    cudaFree(d_count);
    cudaFree(d_hist);

    // Free host memory
    free(h_hist);
    return 0;
}
