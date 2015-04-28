#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>


void allocateMemory(float **d_previousWstep, float **d_random, 
                    int N_v, int N_h){
    int WBYTES = N_v * N_h * sizeof(float);
    checkCudaErrors(cudaMalloc((void **)d_previousWstep, WBYTES));
    checkCudaErrors(cudaMemset(*d_previousWstep, 0, WBYTES));
    checkCudaErrors(cudaMalloc((void **)d_random, sizeof(float)*(N_v+N_h)));
    checkCudaErrors(cudaMemset(*d_random, 0, sizeof(float)*(N_v+N_h)));
}

void freeMemory(float *d_previousWstep, float *d_random){
    cudaFree(d_previousWstep); cudaFree(d_random);
    d_previousWstep = NULL; d_random = NULL;
}
