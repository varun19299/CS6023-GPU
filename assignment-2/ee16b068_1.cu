#include <stdio.h> 

int main() {
    FILE *outfile;
    int nDevices;

    //output file pointer
    outfile = fopen("ee16b068_1.txt", "w");

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        //
        printf("Device Number: %d\n", i);
        //
        printf("  Device name: %s\n", prop.name);
        //
        printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
        //
        printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        //
        printf("  Is L1 Cache supported globally :(0/1) %d\n",prop.globalL1CacheSupported);
        fprintf(outfile,"%d\n",prop.globalL1CacheSupported);
        //
        printf("  Is L1 Cache supported locally :(0/1) %d\n",prop.localL1CacheSupported);
        fprintf(outfile,"%d\n",prop.localL1CacheSupported);
        //
        printf("  L2 Cache Size (bytes) : %d\n",prop.l2CacheSize);
        fprintf(outfile,"%d\n",prop.l2CacheSize);
        //
        printf("  Max no of threads per block : %d\n",prop.maxThreadsPerBlock);
        fprintf(outfile,"%d\n",prop.maxThreadsPerBlock);
        //
        printf("  No of registers available in a block : %d\n",prop.regsPerBlock);
        fprintf(outfile,"%d\n",prop.regsPerBlock);
        //
        printf("  No of registers available in a streaming multiprocessor : %d\n",prop.regsPerMultiprocessor);
        fprintf(outfile,"%d\n",prop.regsPerMultiprocessor);
        //
        printf("  Warp Size :(bytes) %d\n",prop.warpSize);
        fprintf(outfile,"%d\n",prop.warpSize);
        //
        printf("  Grid Size :(bytes) %ld\n",prop.maxGridSize);
        //
        printf("  Total memory :(bytes) %ld\n",prop.totalGlobalMem);
        fprintf(outfile,"%ld\n",prop.totalGlobalMem);
        //
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}

