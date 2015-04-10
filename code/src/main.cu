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

#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define MALLOCSET(arr, bytes, dtype) arr = (dtype *)malloc(bytes);\
					   memset(arr, 0, bytes);
#define CUDAMALLOCSET(arr, bytes) checkCudaErrors(cudaMalloc((void **)&arr, bytes));\
				  checkCudaErrors(cudaMemset(arr, 0, bytes));

void allocateMemory(float **h_W, float **d_W, float **d_random,
		    int N_v, int N_h){
    int WBYTES = N_v * N_h * sizeof(float);
    MALLOCSET(*h_W, WBYTES, float);
    CUDAMALLOCSET(*d_W, WBYTES);
    CUDAMALLOCSET(*d_random, sizeof(float)*(N_v+N_h));
}

void freeMemory(float **h_W, float **d_W,
		float **d_random){
    free(*h_W); *h_W = NULL; 
    cudaFree(d_W); cudaFree(d_random);
    *d_W = NULL; *d_random = NULL;
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

    if (argc < 4){
        printf("Not enough arguments: Usage: ./curbm filename N_visible N_hidden\n");
	return 0;
    }
    else if (argc > 4){
        printf("Too many arguments. Usage: ./curbm filename N_visible N_hidden\n");
        return 0;
    }
    char *filename = argv[1];
    int N_v = atoi(argv[2]);
    int N_h = atoi(argv[3]);
   
    int *spinlist = loadSpins(filename);
/*
    printf("\nN_v = %d, N_h = %d, first spin configuration\n", N_vt, N_ht);
    for (int i=0; i < N_vt; i++){
        printf("%d",spinlist[i]);
    }
*/
    //Time measurement
    cudaEvent_t start, stop;
    float time;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    Layer visible, hidden;

    float *h_W, *d_W, *d_random;
    
    allocateLayer(&visible, N_v);
    allocateLayer(&hidden, N_h);
    allocateMemory(&h_W, &d_W, &d_random, N_v, N_h);
    
    //cuBLAS init
    cublasHandle_t cublasHandle;
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaDeviceSynchronize());
    //cuRAND init
    curandGenerator_t rng;
    checkCudaErrors(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(rng, 920989ULL));
    checkCudaErrors(cudaDeviceSynchronize());
     
    checkCudaErrors(curandGenerateNormal(rng, d_W, (size_t) N_v*N_h, 0.f, 0.05f));
    checkCudaErrors(cudaDeviceSynchronize());
   
    float *h_vtest = (float *)malloc(N_v*sizeof(float)), *d_vtest;
    for (int i=0; i < N_v; i++){
        h_vtest[i] = ((double)rand()/(double)RAND_MAX > 0.5) ? -1.f : 1.f;
    }
    
    checkCudaErrors(cudaMalloc(&d_vtest, N_v*sizeof(float)));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(d_vtest, h_vtest, N_v*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
 
    checkCudaErrors(cudaEventRecord(start, 0));
    computeGibbsSample_vhv(visible, hidden, d_W, d_vtest, d_random, cublasHandle, rng);
    checkCudaErrors(cudaEventRecord(stop, 0));
    
    checkCudaErrors(cudaEventSynchronize(stop));

    cudaEventElapsedTime(&time, start, stop);    
    printf("Elapsed time: %f ms\n", time);

    copyLayerDeviceToHost(&visible);
    copyLayerDeviceToHost(&hidden);
    checkCudaErrors(cudaMemcpy(h_W, d_W, sizeof(float)*N_v*N_h, cudaMemcpyDeviceToHost));


    printf("W = ");
    for (int i=0; i < MIN(N_v,N_h); i++){
        if (i % N_h ==0)
            printf("\n");
	printf("%f\t", h_W[i]);
    }
    printf("\nHidden sample = ");
    for (int j=0; j < N_h; j++){
	if (j % N_v ==0)
	    printf("\n");
	printf("%f\t", hidden.h_sample[j]);
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
	printf("%f\t", visible.h_sample[i]);
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
    
    
    checkCudaErrors(cublasDestroy(cublasHandle));
    checkCudaErrors(curandDestroyGenerator(rng));
    freeLayer(visible); freeLayer(hidden); 
    freeMemory(&h_W, &d_W, &d_random);

    return EXIT_SUCCESS;
}



