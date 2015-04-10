#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>

#include "types.h"

__host__
void copyLayerDeviceToHost(Layer *unitLayer){
    checkCudaErrors(cudaMemcpy(unitLayer->h_sample, unitLayer->d_sample, 
	            unitLayer->BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(unitLayer->h_conditionalP, unitLayer->d_conditionalP, 
	            unitLayer->BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(unitLayer->h_energySum, unitLayer->d_energySum, 
	            unitLayer->BYTES, cudaMemcpyDeviceToHost));
}


__host__
void allocateLayer(Layer *newLayer, int N_units){
    int BYTES = N_units * sizeof(float);
    newLayer->BYTES = BYTES;
    newLayer->N_units = N_units;
    newLayer->h_sample = (float *)malloc(BYTES);
    memset(newLayer->h_sample, 0, BYTES);
    newLayer->h_conditionalP = (float *)malloc(BYTES);
    memset(newLayer->h_conditionalP, 0, BYTES);
    newLayer->h_energySum = (float *)malloc(BYTES);
    memset(newLayer->h_energySum, 0, BYTES);
  
    checkCudaErrors(cudaMalloc((void **)&(newLayer->d_sample), BYTES));
    checkCudaErrors(cudaMemset(newLayer->d_sample, 0, BYTES));
    checkCudaErrors(cudaMalloc((void **)&(newLayer->d_conditionalP), BYTES));
    checkCudaErrors(cudaMemset(newLayer->d_conditionalP, 0, BYTES));
    checkCudaErrors(cudaMalloc((void **)&(newLayer->d_energySum), BYTES));
    checkCudaErrors(cudaMemset(newLayer->d_energySum, 0, BYTES));
}

__host__
void freeLayer(Layer newLayer){
    free(newLayer.h_sample); newLayer.h_sample=NULL;
    cudaFree(newLayer.d_sample); newLayer.d_sample=NULL;
    free(newLayer.h_conditionalP); newLayer.h_conditionalP=NULL;
    cudaFree(newLayer.d_conditionalP); newLayer.d_conditionalP=NULL;
    free(newLayer.h_energySum); newLayer.h_energySum=NULL;
    cudaFree(newLayer.d_energySum); newLayer.d_energySum=NULL;
}


