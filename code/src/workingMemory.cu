#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>


void allocateMemory(float **h_W, float **d_W, 
		    float **h_modelCorrelations, float **d_modelCorrelations,
		    float **h_dataCorrelations, float **d_dataCorrelations,
		    float **d_random, float **d_hiddenRandom,
		    float **d_hiddenGivenData, float **d_hiddenEnergy,
		    int N_v, int N_h){
    int WBYTES = N_v * N_h * sizeof(float);
    int HBYTES = N_h * sizeof(float);
    *h_W = (float *)malloc(WBYTES);
    memset(*h_W, 0, WBYTES);
    //These things can probably go after debugging is done
    *h_modelCorrelations = (float *)malloc(WBYTES);
    memset(*h_modelCorrelations, 0, WBYTES);
    *h_dataCorrelations = (float *)malloc(HBYTES);
    memset(*h_dataCorrelations, 0, HBYTES);

    checkCudaErrors(cudaMalloc((void **)d_W, WBYTES));
    checkCudaErrors(cudaMemset(*d_W, 0, WBYTES));
    checkCudaErrors(cudaMalloc((void **)d_modelCorrelations, WBYTES));
    checkCudaErrors(cudaMemset(*d_modelCorrelations, 0, WBYTES));
    checkCudaErrors(cudaMalloc((void **)d_dataCorrelations, WBYTES));
    checkCudaErrors(cudaMemset(*d_dataCorrelations, 0, WBYTES));
    checkCudaErrors(cudaMalloc((void **)d_hiddenGivenData, HBYTES));
    checkCudaErrors(cudaMemset(*d_hiddenGivenData, 0, HBYTES));
    checkCudaErrors(cudaMalloc((void **)d_hiddenEnergy, HBYTES));
    checkCudaErrors(cudaMemset(*d_hiddenEnergy, 0, HBYTES));

    checkCudaErrors(cudaMalloc((void **)d_random, sizeof(float)*(N_v+N_h)));
    checkCudaErrors(cudaMemset(*d_random, 0, sizeof(float)*(N_v+N_h)));
    checkCudaErrors(cudaMalloc((void **)d_hiddenRandom, HBYTES));
    checkCudaErrors(cudaMemset(*d_hiddenRandom, 0, HBYTES));
}

void freeMemory(float **h_W, float **d_W,
		float **h_modelCorrelations, float **d_modelCorrelations,
		float **h_dataCorrelations, float **d_dataCorrelations,
		float **d_random, float **d_hiddenRandom,
		float **d_hiddenGivenData, float **d_hiddenEnergy){
    free(*h_W); *h_W = NULL;
    free(*h_modelCorrelations), *h_modelCorrelations = NULL; 
    cudaFree(d_W); cudaFree(d_random); cudaFree(d_hiddenRandom);
    cudaFree(d_modelCorrelations); cudaFree(d_hiddenGivenData);
    cudaFree(d_dataCorrelations); cudaFree(d_hiddenEnergy);
    *d_W = NULL; *d_random = NULL; *d_hiddenRandom = NULL;
    *d_modelCorrelations = NULL; d_hiddenGivenData = NULL;
    *d_dataCorrelations = NULL; *d_hiddenEnergy = NULL;
}
