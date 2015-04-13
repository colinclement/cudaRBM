#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>

#include "types.h"

__host__
void copyLayerDeviceToHost(Layer *unitLayer){
    checkCudaErrors(cudaMemcpy(unitLayer->h_samples, unitLayer->d_samples, 
	            unitLayer->SAMPLEBYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(unitLayer->h_conditionalP, unitLayer->d_conditionalP, 
	            unitLayer->BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(unitLayer->h_energySum, unitLayer->d_energySum, 
	            unitLayer->BYTES, cudaMemcpyDeviceToHost));
}


__host__
void allocateLayer(Layer *newLayer, int N_units, int kSamples){
    int BYTES = N_units * sizeof(float);
    newLayer->BYTES = BYTES;
    newLayer->SAMPLEBYTES = BYTES * kSamples;
    newLayer->N_units = N_units;
    newLayer->kSamples = kSamples;
    newLayer->h_samples = (float *)malloc(BYTES * kSamples);
    memset(newLayer->h_samples, 0, BYTES * kSamples);
    newLayer->h_conditionalP = (float *)malloc(BYTES);
    memset(newLayer->h_conditionalP, 0, BYTES);
    newLayer->h_energySum = (float *)malloc(BYTES);
    memset(newLayer->h_energySum, 0, BYTES);
  
    checkCudaErrors(cudaMalloc((void **)&(newLayer->d_samples), newLayer->SAMPLEBYTES));
    checkCudaErrors(cudaMemset(newLayer->d_samples, 0, newLayer->SAMPLEBYTES));

    newLayer->d_samplePtr = newLayer->d_samples; //Start ptr at beginning

    checkCudaErrors(cudaMalloc((void **)&(newLayer->d_conditionalP), BYTES));
    checkCudaErrors(cudaMemset(newLayer->d_conditionalP, 0, BYTES));
    checkCudaErrors(cudaMalloc((void **)&(newLayer->d_energySum), BYTES));
    checkCudaErrors(cudaMemset(newLayer->d_energySum, 0, BYTES));
}

__host__
void freeLayer(Layer newLayer){
    free(newLayer.h_samples); newLayer.h_samples=NULL;
    cudaFree(newLayer.d_samples); newLayer.d_samples=NULL;
    free(newLayer.h_conditionalP); newLayer.h_conditionalP=NULL;
    cudaFree(newLayer.d_conditionalP); newLayer.d_conditionalP=NULL;
    free(newLayer.h_energySum); newLayer.h_energySum=NULL;
    cudaFree(newLayer.d_energySum); newLayer.d_energySum=NULL;
}

__host__
void allocateCorrContainer(DataCorrContainer *container, 
		           int N_v, int N_h, int batchSize){
    int BYTES = N_h * sizeof(float);
    container->BATCHBYTES = N_v * batchSize * sizeof(float);
    int BATCHBYTES = container->BATCHBYTES;
    container->N_v = N_v; container->N_h = N_h; container->batchSize = batchSize;

    checkCudaErrors(cudaMalloc((void **)&(container->d_hiddenRandom), BYTES));
    checkCudaErrors(cudaMemset(container->d_hiddenRandom, 0, BYTES));
    checkCudaErrors(cudaMalloc((void **)&(container->d_hiddenGivenData), BYTES));
    checkCudaErrors(cudaMemset(container->d_hiddenGivenData, 0, BYTES));
    checkCudaErrors(cudaMalloc((void **)&(container->d_hiddenEnergy), BYTES));
    checkCudaErrors(cudaMemset(container->d_hiddenEnergy, 0, BYTES));
    checkCudaErrors(cudaMalloc((void **)&(container->d_visibleBatch), BATCHBYTES));
    checkCudaErrors(cudaMemset(container->d_visibleBatch, 0, BATCHBYTES));
}

__host__
void freeCorrContainer(DataCorrContainer container){
    cudaFree(container.d_hiddenRandom); container.d_hiddenRandom = NULL;
    cudaFree(container.d_hiddenGivenData); container.d_hiddenGivenData = NULL;
    cudaFree(container.d_hiddenEnergy); container.d_hiddenEnergy = NULL;
    cudaFree(container.d_visibleBatch); container.d_visibleBatch = NULL;
}

