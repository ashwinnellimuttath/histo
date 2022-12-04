#include <stdio.h>
#include <stdint.h>

#include "support.h"
#include "kernel.cu"

const unsigned int numStream = 2;

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int *in_h;
    unsigned int* bins_h;
    unsigned int *in_d[numStream];
    unsigned int* bins_d[numStream];
    unsigned int num_elements, num_bins;
    cudaError_t cuda_ret;

    


    cudaStream_t streams[numStream];
    for (int i = 0; i < numStream; i++)
        cudaStreamCreate(&streams[i]);

    printf("\nInitialized..."); fflush(stdout);
    

    if(argc == 1) {
        num_elements = 1000000;
        num_bins = 4096;
    } else if(argc == 2) {
        num_elements = atoi(argv[1]);
        num_bins = 4096;
    } else if(argc == 3) {
        num_elements = atoi(argv[1]);
        num_bins = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./histogram            # Input: 1,000,000, Bins: 4,096"
           "\n    Usage: ./histogram <m>        # Input: m, Bins: 4,096"
           "\n    Usage: ./histogram <m> <n>    # Input: m, Bins: n"
           "\n");
        exit(0);
    }

    const int segmentLenElements = num_elements / numStream;
    const int segmentLenBins = num_bins / numStream;

    initVector(&in_h, num_elements, num_bins);
    // bins_h = (unsigned int*) malloc(num_bins*sizeof(unsigned int));
    cudaHostAlloc((void**)&bins_h, num_bins*sizeof(unsigned int), cudaHostAllocDefault);


    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n    Number of bins = %u\n", num_elements,
        num_bins);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    // cuda_ret = cudaMalloc((void**)&in_d, num_elements * sizeof(unsigned int));
    // if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
    // cuda_ret = cudaMalloc((void**)&bins_d, num_bins * sizeof(unsigned int));
    // if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");


    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            cuda_ret = cudaMalloc((unsigned int**) &in_d[i], sizeof(unsigned int) * segmentLenElements);
            if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
            cuda_ret = cudaMalloc((unsigned int**) &bins_d[i], sizeof(unsigned int) * segmentLenBins);
            if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
        }
        else    // remainder
        {
            cudaMalloc((unsigned int**) &in_d[i], sizeof(unsigned int) * (segmentLenElements + num_elements % numStream));
            cudaMalloc((unsigned int**) &bins_d[i], sizeof(unsigned int) * (segmentLenBins + num_bins % numStream));
        }
    }








    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    // cuda_ret = cudaMemcpy(in_d, in_h, num_elements * sizeof(unsigned int),
    //     cudaMemcpyHostToDevice);
    // if(cuda_ret != cudaSuccess) printf("Unable to copy memory to the device");

    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            cudaMemcpyAsync(in_d[i], in_h + i*segmentLenElements, sizeof(unsigned int)*segmentLenElements, cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(bins_d[i], bins_h + i*segmentLenBins, sizeof(unsigned int)*segmentLenBins, cudaMemcpyHostToDevice, streams[i]);
        }
        else
        {
            cudaMemcpyAsync(in_d[i], in_h + i*segmentLenElements, sizeof(unsigned int)*(segmentLenElements + num_elements % numStream), cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(bins_d[i], bins_h + i*segmentLenBins, sizeof(unsigned int)*(segmentLenBins + num_bins % numStream), cudaMemcpyHostToDevice, streams[i]);
        }
    }






    cuda_ret = cudaMemset(bins_d, 0, num_bins * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) printf("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    // histogram(in_d, bins_d, num_elements, num_bins);

    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            histogram(in_d[i], bins_d[i], segmentLenElements, segmentLenBins, streams[i]);
        }
        else
        {
            // histogram(in_d[i], bins_d[i], C_d[i], segmentLen + VecSize % numStream, streams[i]);
            histogram(in_d[i], bins_d[i], segmentLenElements + num_elements % numStream, segmentLenBins + num_bins % numStream, streams[i]);
        }
    }










    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    // cuda_ret = cudaMemcpy(bins_h, bins_d, num_bins * sizeof(unsigned int),
    //     cudaMemcpyDeviceToHost);
	//   if(cuda_ret != cudaSuccess) printf("Unable to copy memory to host");

    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            cudaMemcpyAsync(bins_h + i*segmentLenBins, bins_d[i], sizeof(unsigned int)*segmentLenBins, cudaMemcpyDeviceToHost, streams[i]);
        }
        else
        {
            cudaMemcpyAsync(bins_h + i*segmentLenBins, bins_d[i], sizeof(unsigned int)*(segmentLenBins + num_bins % numStream), cudaMemcpyDeviceToHost, streams[i]);
        }
    }








    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, bins_h, num_elements, num_bins);

    // Free memory ------------------------------------------------------------

    // cudaFree(in_d); cudaFree(bins_d);
    // free(in_h); free(bins_h);
    cudaFreeHost(in_h);
    cudaFreeHost(bins_h);

    for (int i = 0; i < numStream; i++)
    {
        cudaFree(in_d[i]);
        cudaFree(bins_d[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}

