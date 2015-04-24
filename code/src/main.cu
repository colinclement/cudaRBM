#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <helper_cuda.h> //In samples/common/inc

#include "loadSpins.h"
#include "layers.h"
#include "sampleCorrelate.h"
#include "workingMemory.h"
#include "energyFunctions.h"
#include "types.h"

//#define DBUG //Save stuff to files

#define INVSIGN(x) ((x > 0) ? -1.f: 1.f)
#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define IDX2F(i,j,ld) (((j)*(ld))+(i))

#define THREADS_PER 256

__global__
void weightMatrixUpdate(float *d_W, float *d_previousWstep, 
		        float *d_modelCorrelations, float *d_dataCorrelations, 
			float lr, float mom, float sparsity,
			int batchSize, int N_v, int N_h);

int main(int argc, char **argv){

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0){
        fprintf(stderr, "Error: no CUDA supporting devices.\n");
	exit(EXIT_FAILURE);
    }
    int dev = 0; 
    cudaSetDevice(dev);
    
    const char *printMSG = "Incorrect number of arguments: Usage: \n\
			    ./curbm filename N_visible N_hidden k_samples batchsize epochs lr mom sparsity\n";
    if (argc < 10){
        printf("%s", printMSG);
	return 0;
    }
    else if (argc > 10){
        printf("%s", printMSG);
        return 0;
    }

    char *filename = argv[1];
    int N_v = atoi(argv[2]);
    int N_h = atoi(argv[3]);
    int k = atoi(argv[4]);
    int batchSize = atoi(argv[5]);
    int epochs = atoi(argv[6]);
    float lr = atof(argv[7]);
    float mom = atof(argv[8]);
    float sparsity = atof(argv[9]);
    printf("Learning rate = %f, momemtum = %f, sparsity = %f\n", lr, mom, sparsity);

    int Nbits = 0, numSamples = 0;
    float *h_spinList = loadSpins(filename, &Nbits);
    if (h_spinList == NULL){
        printf("Exiting.\n");
	return 0;	   
    }
    numSamples = Nbits / N_v;

    Layer visibleModel = allocateLayer(N_v, k);
    Layer hiddenModel = allocateLayer(N_h, k);
    Layer visibleData = allocateLayer(N_v, batchSize);
    Layer hiddenData = allocateLayer(N_h, batchSize);
    //DataCorrContainer container;
    float *h_W, *d_W, *d_previousWstep;
    float *h_modelCorrelations, *d_modelCorrelations, *d_random;
    float *h_dataCorrelations, *d_dataCorrelations;
    allocateMemory(&h_W, &d_W, &d_previousWstep,
		   &h_modelCorrelations, &d_modelCorrelations,
		   &h_dataCorrelations, &d_dataCorrelations,
		   &d_random, N_v, N_h);
    
    cudaStream_t stream1, stream2;
    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaStreamCreate(&stream2)); 
    //cuBLAS init
    cublasHandle_t cublasHandle1, cublasHandle2;
    checkCudaErrors(cublasCreate(&cublasHandle1)); checkCudaErrors(cublasCreate(&cublasHandle2));
    checkCudaErrors(cublasSetStream(cublasHandle1, stream1));
    checkCudaErrors(cublasSetStream(cublasHandle2, stream2));
    //cuRAND init
    curandGenerator_t rng1, rng2;
    checkCudaErrors(curandCreateGenerator(&rng1, CURAND_RNG_PSEUDO_DEFAULT));
    checkCudaErrors(curandSetStream(rng1, stream1));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(rng1, 920989ULL));
    checkCudaErrors(curandCreateGenerator(&rng2, CURAND_RNG_PSEUDO_DEFAULT));
    checkCudaErrors(curandSetStream(rng2, stream2));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(rng2, 14859ULL));
    
    //initialize Weights
    float stdDev = 1.f/sqrt( (float) N_h); 
    checkCudaErrors(curandGenerateNormal(rng1, d_W, (size_t) N_v*N_h, 0.f, stdDev));
    //Copy initial weights to device
    checkCudaErrors(cudaMemcpy(h_W, d_W, sizeof(float)*N_v*N_h, cudaMemcpyDeviceToHost));

    //Time measurement
    cudaEvent_t start, stop;
    float time, stepL1;
    checkCudaErrors(cudaEventCreate(&start)); 
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    dim3 blocks((int) ceil((float) (N_v * N_h)/(float) THREADS_PER), 1, 1);
    dim3 threads(THREADS_PER, 1, 1);
    int numBatches = (int) ceil((float) numSamples / (float) batchSize);
    FILE *fpConv = fopen("Convergence.dat","w");
    
    //epochs = 1; numBatches = 4; //For NVVP single iterations
    //TODO: Implement PCG RNG for improved performance

    printf("Performing %d epochs with  %d batches\n", epochs, numBatches);
    for (int ep = 0; ep < epochs; ep++){
        for (int i = 0; i < numBatches; i++){
        checkCudaErrors(cudaDeviceSynchronize());
        int randIndex = (int)( ((float)rand()/(RAND_MAX))*numSamples);
        int startGibbs = MAX(1, N_v * MIN(numSamples-1, randIndex)) - 1;
        updateLayerSample(visibleModel, &(h_spinList[startGibbs]),
                          visibleModel.BYTES, stream1); 
        
        float *h_batchPtr = &(h_spinList[N_v*MIN(batchSize*i, 
                                                 numSamples-batchSize-1)]);
        updateLayerSample(visibleData, h_batchPtr, 
                          visibleData.SAMPLEBYTES, stream2);

        computeKGibbs(visibleModel, hiddenModel, d_W, allToAll, 
                      d_random, rng1, stream1, cublasHandle1);
        computeCorrelations(visibleModel, hiddenModel, d_modelCorrelations,
                            cublasHandle1);

        computeGibbsGivenData(visibleData, hiddenData, d_W, allToAll,
                              rng2, stream2, cublasHandle2);
        computeCorrelations(visibleData, hiddenData, d_dataCorrelations,
                            cublasHandle2);

        checkCudaErrors(cublasSasum(cublasHandle1, N_h*N_v, d_previousWstep, 1, &stepL1));
	    fprintf(fpConv, "%f\n", stepL1);
        //Update weights
        checkCudaErrors(cudaStreamSynchronize(stream1)); 
        checkCudaErrors(cudaStreamSynchronize(stream2));

        weightMatrixUpdate<<<blocks, threads,
                             0,    stream1>>>(d_W, d_previousWstep,
 	                                          d_modelCorrelations, d_dataCorrelations, 
                                              lr, mom, sparsity, batchSize, N_v, N_h);
        }
    }
    checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());
  
    //Stop timer
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    cudaEventElapsedTime(&time, start, stop);    
    printf("Elapsed time: %f ms\n", time);

    FILE *fp_saveW = fopen("W.dat", "w");
    //Save initial weights
    for (int i=0; i < N_v; i++){
        fprintf(fp_saveW, "\n");
        for (int j=0; j < N_h; j++){
            fprintf(fp_saveW, "%f\t", h_W[IDX2F(i,j, N_v)]);
        }
    } 
    checkCudaErrors(cudaMemcpy(h_W, d_W, sizeof(float)*N_v*N_h, cudaMemcpyDeviceToHost));
    // Save final weights 
    for (int i=0; i < N_v; i++){
        fprintf(fp_saveW, "\n");
        for (int j=0; j < N_h; j++){
            fprintf(fp_saveW, "%f\t", h_W[IDX2F(i,j, N_v)]);
        }
    }
    fclose(fp_saveW);

#ifdef DBUG
 
    copyLayerDeviceToHost(visibleModel);
    copyLayerDeviceToHost(hiddenModel);
    copyLayerDeviceToHost(visibleData);
    copyLayerDeviceToHost(hiddenData);
    checkCudaErrors(cudaMemcpy(h_modelCorrelations, d_modelCorrelations, 
			       sizeof(float)*N_v*N_h, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_dataCorrelations, d_dataCorrelations, 
			       sizeof(float)*N_v*N_h, cudaMemcpyDeviceToHost));
    
    FILE *fpW = fopen("dbugW.dat", "w");
    FILE *fph = fopen("dbugHidden.dat", "w");
    FILE *fpv = fopen("dbugVisible.dat", "w");
    FILE *fpvcorr = fopen("dbugDcorrV.dat", "w");
    FILE *fphcorr = fopen("dbugDcorrH.dat", "w");
   
    //For data correlations: 
    for (int j=0; j < N_v * visibleData.numSamples; j++){
        if (j % N_v == 0)
            fprintf(fpvcorr, "\n");
        fprintf(fpvcorr, "%f\t", visibleData.h_samples[j]);
    }
    fclose(fpvcorr);
    for (int j=0; j < N_h; j++){
        if (j % N_h == 0)
            fprintf(fphcorr, "\n");
        fprintf(fphcorr, "%f\t", hiddenData.h_random[j]);
    }
    for (int j=0; j < N_h; j++){
        if (j % N_h == 0)
            fprintf(fphcorr, "\n");
        fprintf(fphcorr, "%f\t", hiddenData.h_energySum[j]);
    }
    for (int j=0; j < N_h * hiddenData.numSamples; j++){
        if (j % N_h == 0)
            fprintf(fphcorr, "\n");
        fprintf(fphcorr, "%f\t", hiddenData.h_samples[j]);
    }
    fclose(fphcorr);
    //For model correlations:
    for (int i=0; i < N_v; i++){
        fprintf(fpW, "\n");
        for (int j=0; j < N_h; j++){
            fprintf(fpW, "%f\t", h_W[IDX2F(i,j, N_v)]);
        }
    }
    for (int i=0; i < N_v; i++){
        fprintf(fpW, "\n");
        for (int j=0; j < N_h; j++){
            fprintf(fpW, "%f\t", h_modelCorrelations[IDX2F(i,j, N_v)]);
        }
    }
    for (int i=0; i < N_v; i++){
        fprintf(fpW, "\n");
        for (int j=0; j < N_h; j++){
            fprintf(fpW, "%f\t", h_dataCorrelations[IDX2F(i,j, N_v)]);
        }
    }
    for (int j=0; j < N_h; j++){
 	if (j % N_h ==0)
 	    fprintf(fph, "\n");
 	fprintf(fph, "%f\t", hiddenModel.h_conditionalP[j]);
    }
    for (int j=0; j < N_h; j++){
	if (j % N_h ==0)
	    fprintf(fph, "\n");
	fprintf(fph, "%f\t", hiddenModel.h_energySum[j]);
    }
    int nhiddens = hiddenModel.numSamples * N_h;
    for (int j=0; j < nhiddens; j++){
	if (j % N_h ==0)
	    fprintf(fph, "\n");
	fprintf(fph, "%f\t", hiddenModel.h_samples[j]);
    }
    for (int i=0; i < N_v; i++){
	if (i % N_v == 0){
	    fprintf(fpv, "\n");
	}
	fprintf(fpv, "%f\t", visibleModel.h_conditionalP[i]);
    }
    for (int i=0; i < N_v; i++){
	if (i % N_v == 0){
	    fprintf(fpv, "\n");
	}
	fprintf(fpv, "%f\t", visibleModel.h_energySum[i]);
    }
    for (int i=0; i < N_v; i++){
	if (i % N_v == 0)
	    fprintf(fpv, "\n");
        fprintf(fpv, "%f\t", h_spinList[i]);
    }
    int nvisibles = visibleModel.numSamples * N_v;
    for (int i=0; i < nvisibles; i++){
	if (i % N_v == 0){
	    fprintf(fpv, "\n");
	}
	fprintf(fpv, "%f\t", visibleModel.h_samples[i]);
    }
    fclose(fpW);
    fclose(fph);
    fclose(fpv);

#endif
   
    // Clean up 
    checkCudaErrors(cublasDestroy(cublasHandle1));
    checkCudaErrors(cublasDestroy(cublasHandle2));
    checkCudaErrors(curandDestroyGenerator(rng1));
    checkCudaErrors(curandDestroyGenerator(rng2));
    checkCudaErrors(cudaStreamDestroy(stream1));
    checkCudaErrors(cudaStreamDestroy(stream2));

    freeLayer(visibleModel); freeLayer(hiddenModel);
    freeLayer(visibleData); freeLayer(hiddenData);
    freeMemory(h_W, d_W, d_previousWstep, 
	       h_modelCorrelations, d_modelCorrelations,
	       h_dataCorrelations, d_dataCorrelations,
	       d_random);
    checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());

    return EXIT_SUCCESS;
}

__global__
void weightMatrixUpdate(float *d_W, float *d_previousWstep,
		        float *d_modelCorrelations,
		        float *d_dataCorrelations, 
			float lr, float mom, float sparsity,
			int batchSize, int N_v, int N_h){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_v * N_h){
        return;
    }
    float W = d_W[tid];
    float lastStep = d_previousWstep[tid];
    float corrDiff = d_dataCorrelations[tid] - d_modelCorrelations[tid];
    float CDstep = (lr / ((float) batchSize)) * corrDiff;
    float newStep = (1.f-mom)*CDstep + mom*lastStep + sparsity*INVSIGN(W);
    d_previousWstep[tid] = newStep; //update previous steps
    d_W[tid] = W + newStep; 
}


