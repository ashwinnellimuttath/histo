#include <stdio.h>
#define TILE_SIZE 512
#define BLOCK_SIZE 512
__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
	__shared__ unsigned int hist_private[5000];
    //extern __shared__ unsigned int shared_bins[];
    //unsigned int *hist_private = &shared_bins[0];
    for (int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE)
        hist_private[i] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < num_elements)
    {
        atomicAdd(&(hist_private[input[i]]), 1);
        i += stride;
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE)
        atomicAdd(&(bins[i]), hist_private[i]);	
	
	  /*************************************************************************/
}

void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

	  /*************************************************************************/
    //INSERT CODE HERE
    //const unsigned int BLOCK_SIZE = TILE_SIZE;

    dim3 DimGrid((num_elements-1)/BLOCK_SIZE + 1,1,1); 
    dim3 DimBlock(BLOCK_SIZE,1);

    histo_kernel<<<DimGrid, DimBlock>>>(input, bins, num_elements, num_bins);

	  /*************************************************************************/

}


