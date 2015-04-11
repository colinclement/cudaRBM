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
#include "types.h"

#define DBUG
//#define DBUG_LOAD

#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

void allocateMemory(float **h_W, float **d_W, 
		    float **h_modelCorrelations, float **d_modelCorrelations,
		    float **d_random,
		    int N_v, int N_h){
    int WBYTES = N_v * N_h * sizeof(float);
    *h_W = (float *)malloc(WBYTES);
    memset(*h_W, 0, WBYTES);
    *h_modelCorrelations = (float *)malloc(WBYTES);
    memset(*h_modelCorrelations, 0, WBYTES);
    checkCudaErrors(cudaMalloc((void **)d_W, WBYTES));
    checkCudaErrors(cudaMemset(*d_W, 0, WBYTES));
    checkCudaErrors(cudaMalloc((void **)d_modelCorrelations, WBYTES));
    checkCudaErrors(cudaMemset(*d_modelCorrelations, 0, WBYTES));
    checkCudaErrors(cudaMalloc((void **)d_random, sizeof(float)*(N_v+N_h)));
    checkCudaErrors(cudaMemset(*d_W, 0, sizeof(float)*(N_v+N_h)));
}

void freeMemory(float **h_W, float **d_W,
		float **h_modelCorrelations, float **d_modelCorrelations,
		float **d_random){
    free(*h_W); *h_W = NULL;
    free(*h_modelCorrelations), *h_modelCorrelations = NULL; 
    cudaFree(d_W); cudaFree(d_random);
    cudaFree(d_modelCorrelations);
    *d_W = NULL; *d_random = NULL;
    *d_modelCorrelations = NULL;
}

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
			    ./curbm filename N_visible N_hidden k_samples\n";
    if (argc < 5){
        printf("%s", printMSG);
	return 0;
    }
    else if (argc > 5){
        printf("%s", printMSG);
        return 0;
    }
    char *filename = argv[1];
    int N_v = atoi(argv[2]);
    int N_h = atoi(argv[3]);
    int k = atoi(argv[4]);
  
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
     
#ifdef DBUG
    printf("Allocating layers\n");
#endif

    allocateLayer(&visible, N_v, k);
    allocateLayer(&hidden, N_h, k);
    allocateMemory(&h_W, &d_W, 
		   &h_modelCorrelations, &d_modelCorrelations,
		   &d_random, N_v, N_h);
    
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

    float *d_initialVisible;
    checkCudaErrors(cudaMalloc(&d_initialVisible, visible.BYTES));
    //Move over first config to device
    checkCudaErrors(cudaMemcpy(d_initialVisible, h_spinlist, visible.BYTES, cudaMemcpyHostToDevice));

#ifdef DBUG_LOAD
    printf("visible.BYTES = %d\n", visible.BYTES);
    printf("first spin configuration:\n");
    for (int i=0; i < N_v; i++){
	if (i % 8 == 0)
	    printf("\n");
        printf("%d\t", (int) h_spinlist[i]);
    } printf("\n");
#endif 

    checkCudaErrors(cudaEventRecord(start, 0));
    computeK_Gibbs(visible, hidden, d_W, d_initialVisible, d_random, cublasHandle, rng);
    computeModelCorrelations(visible, hidden, d_modelCorrelations, cublasHandle);
    checkCudaErrors(cudaEventRecord(stop, 0));
    
    checkCudaErrors(cudaEventSynchronize(stop));

    cudaEventElapsedTime(&time, start, stop);    
    printf("Elapsed time: %f ms\n", time);

    copyLayerDeviceToHost(&visible);
    copyLayerDeviceToHost(&hidden);
    checkCudaErrors(cudaMemcpy(h_modelCorrelations, d_modelCorrelations, 
			       sizeof(float)*N_v*N_h, cudaMemcpyDeviceToHost));

#ifdef DBUG
    printf("model correlations = ");
    for (int i=0; i < N_v*N_h; i++){
        if (i % N_h ==0)
            printf("\n");
	printf("%f\t", h_modelCorrelations[i]);
    }
    printf("\nHidden sample = ");
    for (int j=0; j < N_h; j++){
	if (j % N_v ==0)
	    printf("\n");
	printf("%f\t", hidden.h_samples[j]);
    }
    printf("\nHidden Conditional Probability = ");
    for (int j=0; j < N_h; j++){
	if (j % N_v ==0)
	    printf("\n");
	printf("%f\t", hidden.h_conditionalP[j]);
    }
    printf("\nHidden Energies = ");
    for (int j=0; j < N_h; j++){
	if (j % N_v ==0)
	    printf("\n");
	printf("%f\t", hidden.h_energySum[j]);
    }
    printf("\nVisible sample = ");
    for (int i=0; i < N_h; i++){
	if (i % N_h == 0){
	    printf("\n");
	}
	printf("%f\t", visible.h_samples[i]);
    }
    printf("\nVisible Conditional Probability = ");
    for (int i=0; i < N_h; i++){
	if (i % N_h == 0){
	    printf("\n");
	}
	printf("%f\t", visible.h_conditionalP[i]);
    }
    printf("\nVisible energies = ");
    for (int i=0; i < N_h; i++){
	if (i % N_h == 0){
	    printf("\n");
	}
	printf("%f\t", visible.h_energySum[i]);
    }
    printf("\n");
#endif
    
    checkCudaErrors(cublasDestroy(cublasHandle));
    checkCudaErrors(curandDestroyGenerator(rng));
    freeLayer(visible); freeLayer(hidden); 
    freeMemory(&h_W, &d_W, &h_modelCorrelations, &d_modelCorrelations,
	       &d_random);

    return EXIT_SUCCESS;
}



