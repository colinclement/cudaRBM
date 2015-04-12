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
#include "types.h"

#define DBUG
#define DBUG_LOAD

#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define IDX2F(i,j,ld) (((j)*(ld))+(i))


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
			    ./curbm filename N_visible N_hidden k_samples batchsize\n";
    if (argc < 6){
        printf("%s", printMSG);
	return 0;
    }
    else if (argc > 6){
        printf("%s", printMSG);
        return 0;
    }
    char *filename = argv[1];
    int N_v = atoi(argv[2]);
    int N_h = atoi(argv[3]);
    int k = atoi(argv[4]);
    int batchSize = atoi(argv[5]);
  
    int Nbits = 0;//, numSamples = 0;

#ifdef DBUG_LOAD
    printf("Loading spins from %s\n", filename);
#endif

    float *h_spinlist = loadSpins(filename, &Nbits);
    if (h_spinlist == NULL){
        printf("Exiting.\n");
	return 0;	   
    }
    //numSamples = Nbits / N_v;

#ifdef DBUG_LOAD
    printf("Spins loaded!\n");
#endif

    Layer visible, hidden;

    float *h_W, *d_W;
    float *h_modelCorrelations, *d_modelCorrelations, *d_random;
    float *h_dataCorrelations, *d_dataCorrelations;
    //For Sampling data distribution
    float *d_hiddenGivenData, *d_hiddenEnergy, *d_hiddenRandom;
     
#ifdef DBUG
    printf("Allocating layers\n");
#endif

    allocateLayer(&visible, N_v, k);
    allocateLayer(&hidden, N_h, k);
    allocateMemory(&h_W, &d_W,
		   &h_modelCorrelations, &d_modelCorrelations,
		   &h_dataCorrelations, &d_dataCorrelations,
		   &d_random, &d_hiddenRandom,
		   &d_hiddenGivenData, &d_hiddenEnergy,
		   N_v, N_h);
    
    //cuBLAS init
    cublasHandle_t cublasHandle;
    checkCudaErrors(cublasCreate(&cublasHandle));
    //cuRAND init
    curandGenerator_t rng;
    checkCudaErrors(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(rng, 920989ULL));
    //initialize Weights 
    checkCudaErrors(curandGenerateNormal(rng, d_W, (size_t) N_v*N_h, 0.f, 0.05f));
   

   
    //Time measurement
    cudaEvent_t start, stop;
    float time;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
/*
    float *h_vtest = (float *)malloc(N_v*sizeof(float)), *d_vtest;
    for (int i=0; i < N_v; i++){
        h_vtest[i] = ((double)rand()/(double)RAND_MAX > 0.5) ? -1.f : 1.f;
    }
    
    checkCudaErrors(cudaMalloc(&d_vtest, N_v*sizeof(float)));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(d_vtest, h_vtest, N_v*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
*/

    float *d_initialVisible, *d_batch;
    checkCudaErrors(cudaMalloc(&d_initialVisible, visible.BYTES));
    checkCudaErrors(cudaMalloc(&d_batch, visible.BYTES * batchSize)); 
    checkCudaErrors(cudaMemcpy(d_initialVisible, h_spinlist, visible.BYTES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_batch, h_spinlist, visible.BYTES * batchSize, cudaMemcpyHostToDevice));

    FILE *fpW = fopen("testW.dat", "w");
    FILE *fph = fopen("testhidden.dat", "w");
    FILE *fpv = fopen("testvisible.dat", "w");

#ifdef DBUG_LOAD
    printf("visible.BYTES = %d\n", visible.BYTES);
    printf("first spin configuration:\n");
    for (int i=0; i < N_v; i++){
	if (i % N_v == 0)
	    fprintf(fpv, "\n");
        fprintf(fpv, "%f\t", h_spinlist[i]);
    } 
#endif 

    checkCudaErrors(cudaEventRecord(start, 0));
    computeK_Gibbs(visible, hidden, d_W, d_initialVisible, d_random, cublasHandle, rng);
    computeModelCorrelations(visible, hidden, d_modelCorrelations, cublasHandle);
    computeDataCorrelations(d_dataCorrelations, d_W, d_batch, d_hiddenRandom, d_hiddenGivenData, 
		            d_hiddenEnergy, N_v, N_h, batchSize, cublasHandle, rng);
    checkCudaErrors(cudaEventRecord(stop, 0));
    
    checkCudaErrors(cudaEventSynchronize(stop));

    cudaEventElapsedTime(&time, start, stop);    
    printf("Elapsed time: %f ms\n", time);

    copyLayerDeviceToHost(&visible);
    copyLayerDeviceToHost(&hidden);
    checkCudaErrors(cudaMemcpy(h_modelCorrelations, d_modelCorrelations, 
			       sizeof(float)*N_v*N_h, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_W, d_W, sizeof(float)*N_v*N_h, cudaMemcpyDeviceToHost));


#ifdef DBUG
    //printf("model correlations = ");
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
    //printf("\nHidden sample = ");
    int nhiddens = hidden.kSamples * N_h;
    for (int j=0; j < nhiddens; j++){
	if (j % N_h ==0)
	    fprintf(fph, "\n");
	fprintf(fph, "%f\t", hidden.h_samples[j]);
    }
    //printf("\nHidden Conditional Probability = ");
    for (int j=0; j < N_h; j++){
	if (j % N_h ==0)
	    fprintf(fph, "\n");
	fprintf(fph, "%f\t", hidden.h_conditionalP[j]);
    }
    //printf("\nHidden Energies = ");
    for (int j=0; j < N_h; j++){
	if (j % N_h ==0)
	    fprintf(fph, "\n");
	fprintf(fph, "%f\t", hidden.h_energySum[j]);
    }
    int nvisibles = visible.kSamples * N_v;
    //printf("\nVisible sample = ");
    for (int i=0; i < nvisibles; i++){
	if (i % N_v == 0){
	    fprintf(fpv, "\n");
	}
	fprintf(fpv, "%f\t", visible.h_samples[i]);
    }
    //printf("\nVisible Conditional Probability = ");
    for (int i=0; i < N_v; i++){
	if (i % N_v == 0){
	    fprintf(fpv, "\n");
	}
	fprintf(fpv, "%f\t", visible.h_conditionalP[i]);
    }
    //printf("\nVisible energies = ");
    for (int i=0; i < N_v; i++){
	if (i % N_v == 0){
	    fprintf(fpv, "\n");
	}
	fprintf(fpv, "%f\t", visible.h_energySum[i]);
    }
    fclose(fpW);
    fclose(fph);
    fclose(fpv);

    //printf("\n");
#endif
    
    checkCudaErrors(cublasDestroy(cublasHandle));
    checkCudaErrors(curandDestroyGenerator(rng));

    freeLayer(visible); freeLayer(hidden); 
    freeMemory(&h_W, &d_W, 
	       &h_modelCorrelations, &d_modelCorrelations,
	       &h_dataCorrelations, &d_dataCorrelations,
	       &d_random, &d_hiddenRandom, 
	       &d_hiddenGivenData, &d_hiddenEnergy);

    return EXIT_SUCCESS;
}



