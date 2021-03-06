#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <curand.h>
#include "types.h"

__host__
void copyLayerDeviceToHost(Layer unitLayer){
    checkCudaErrors(cudaMemcpy(unitLayer.h_samples, unitLayer.d_samples, 
	            unitLayer.SAMPLEBYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(unitLayer.h_conditionalP, unitLayer.d_conditionalP, 
	            unitLayer.BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(unitLayer.h_energySum, unitLayer.d_energySum, 
	            unitLayer.BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(unitLayer.h_random, unitLayer.d_random, 
	            unitLayer.BYTES, cudaMemcpyDeviceToHost));
}

__host__
Layer allocateLayer(int N_units, int numSamples){
    Layer newLayer;
    int BYTES = N_units * sizeof(float);
    newLayer.BYTES = BYTES;
    newLayer.SAMPLEBYTES = BYTES * numSamples;
    newLayer.N_units = N_units;
    newLayer.numSamples = numSamples;
    newLayer.h_samples = (float *)malloc(BYTES * numSamples);
    memset(newLayer.h_samples, 0, BYTES * numSamples);
    newLayer.h_conditionalP = (float *)malloc(BYTES);
    memset(newLayer.h_conditionalP, 0, BYTES);
    newLayer.h_energySum = (float *)malloc(BYTES);
    memset(newLayer.h_energySum, 0, BYTES);
    newLayer.h_random = (float *)malloc(BYTES);
    memset(newLayer.h_random, 0, BYTES);
  
    checkCudaErrors(cudaMalloc((void **)&newLayer.d_samples, newLayer.SAMPLEBYTES));
    checkCudaErrors(cudaMemset(newLayer.d_samples, 0, newLayer.SAMPLEBYTES));

    newLayer.d_samplePtr = newLayer.d_samples; //Start ptr at beginning

    checkCudaErrors(cudaMalloc((void **)&newLayer.d_random, BYTES));
    checkCudaErrors(cudaMemset(newLayer.d_random, 0, BYTES));
    checkCudaErrors(cudaMalloc((void **)&newLayer.d_conditionalP, BYTES));
    checkCudaErrors(cudaMemset(newLayer.d_conditionalP, 0, BYTES));
    checkCudaErrors(cudaMalloc((void **)&newLayer.d_energySum, BYTES));
    checkCudaErrors(cudaMemset(newLayer.d_energySum, 0, BYTES));

    return newLayer;
}

__host__
Connection allocateConnection(int N_v, int N_h, curandGenerator_t rng,
                              Filter filterType, int width, int stride){ 
    Connection newConn;
    newConn.fan_in = N_v; newConn.fan_out = N_h;
    newConn.filterType = filterType;
    switch(filterType){
        case ALLTOALL:
            newConn.rows = N_v; newConn.cols = N_h;
            newConn.width = 0; newConn.stride = 1;
            break;
        case CONVOLUTION:
            newConn.rows = width; newConn.cols = width;
            newConn.width = width; newConn.stride = stride;
            break;
        default:
            break;
    }
    newConn.FILTERBYTES = newConn.rows * newConn.cols * sizeof(float);
    newConn.CORRBYTES = sizeof(float) * N_v * N_h;
    int FBYTES = newConn.FILTERBYTES;
    int CBYTES = newConn.CORRBYTES;

    newConn.h_W = (float *)malloc(FBYTES);
    memset(newConn.h_W, 0, FBYTES);
    newConn.h_modelCorrelations = (float *)malloc(CBYTES);
    memset(newConn.h_modelCorrelations, 0, CBYTES);
    newConn.h_dataCorrelations = (float *)malloc(CBYTES);
    memset(newConn.h_dataCorrelations, 0, CBYTES);
    
    checkCudaErrors(cudaMalloc((void **)&newConn.d_W, FBYTES));
    checkCudaErrors(cudaMemset(newConn.d_W, 0, FBYTES));
    checkCudaErrors(cudaMalloc((void **)&newConn.d_modelCorrelations, CBYTES));
    checkCudaErrors(cudaMemset(newConn.d_modelCorrelations, 0, CBYTES));
    checkCudaErrors(cudaMalloc((void **)&newConn.d_dataCorrelations, CBYTES));
    checkCudaErrors(cudaMemset(newConn.d_dataCorrelations, 0, CBYTES));

    float stdDev = 1.f/sqrt( (float) newConn.cols); 
    curandGenerateNormal(rng, newConn.d_W, (size_t) newConn.rows*newConn.cols, 
                         0.f, stdDev);
    //Copy initial weights to device
    checkCudaErrors(cudaMemcpy(newConn.h_W, newConn.d_W, 
                               FBYTES, cudaMemcpyDeviceToHost));

    return newConn;
}

__host__
void updateLayerSample(Layer unitLayer, float *h_hostSamples, int  BYTES,
                       cudaStream_t stream){
    checkCudaErrors(cudaMemcpyAsync(unitLayer.d_samples, h_hostSamples, BYTES,
                                    cudaMemcpyHostToDevice, stream));
}

__host__
void freeLayer(Layer newLayer){
    free(newLayer.h_samples); newLayer.h_samples=NULL;
    cudaFree(newLayer.d_samples); newLayer.d_samples=NULL;
    free(newLayer.h_conditionalP); newLayer.h_conditionalP=NULL;
    cudaFree(newLayer.d_random); newLayer.d_random=NULL;
    cudaFree(newLayer.d_conditionalP); newLayer.d_conditionalP=NULL;
    free(newLayer.h_energySum); newLayer.h_energySum=NULL;
    cudaFree(newLayer.d_energySum); newLayer.d_energySum=NULL;
    free(newLayer.h_random); newLayer.h_random = NULL;
}

__host__
void freeConnection(Connection newConn){
    free(newConn.h_W); newConn.h_W = NULL;
    cudaFree(newConn.d_W); newConn.d_W = NULL;
    free(newConn.h_modelCorrelations);
    newConn.h_modelCorrelations = NULL;
    cudaFree(newConn.d_modelCorrelations);
    newConn.d_modelCorrelations = NULL;
    free(newConn.h_dataCorrelations);
    newConn.h_dataCorrelations = NULL;
    cudaFree(newConn.d_dataCorrelations);
    newConn.d_dataCorrelations = NULL;
}


