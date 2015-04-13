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

//#define DBUG //Save stuff to files

#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define IDX2F(i,j,ld) (((j)*(ld))+(i))

#define THREADS_PER 64

__global__
void weightMatrixUpdate(float *d_W, float *d_modelCorrelations,
		        float *d_dataCorrelations, 
			float lr, float mom, float sparsity,
			int N_h, int N_v);

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
  
    int Nbits = 0, numSamples = 0;
    float *h_spinList = loadSpins(filename, &Nbits);
    if (h_spinList == NULL){
        printf("Exiting.\n");
	return 0;	   
    }
    numSamples = Nbits / N_v;

    Layer visible, hidden;
    DataCorrContainer container;
    float *h_W, *d_W;
    float *h_modelCorrelations, *d_modelCorrelations, *d_random;
    float *h_dataCorrelations, *d_dataCorrelations;
    allocateLayer(&visible, N_v, k);
    allocateLayer(&hidden, N_h, k);
    allocateCorrContainer(&container, N_v, N_h, batchSize);
    allocateMemory(&h_W, &d_W,
		   &h_modelCorrelations, &d_modelCorrelations,
		   &h_dataCorrelations, &d_dataCorrelations,
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
    checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&stop));

    float *d_initialVisible, *h_spinPtr = h_spinList;
    checkCudaErrors(cudaMalloc(&d_initialVisible, visible.BYTES));

    //Start timer
    checkCudaErrors(cudaEventRecord(start, 0));

    dim3 blocks(ceil((float) (N_v * N_h)/(float) THREADS_PER), 1, 1);
    dim3 threads(THREADS_PER, 1, 1);
    int numBatches = ceil((float) numSamples / (float) batchSize);

    printf("Only doing 5 batches for profiling\n");
    for (int ep = 0; ep < epochs; ep++){
        for (int i = 0; i < 1; i++){ 
            int startGibbs = ceil((rand()/(float)RAND_MAX) * numSamples);
            checkCudaErrors(cudaMemcpy(d_initialVisible, h_spinList + N_v*startGibbs, 
        			           visible.BYTES, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(container.d_visibleBatch, h_spinPtr, visible.BYTES * batchSize, 
            			   cudaMemcpyHostToDevice));
            h_spinPtr += MIN(N_v * batchSize, numSamples - batchSize - 1);
            
            computeK_Gibbs(visible, hidden, d_W, d_initialVisible, d_random, cublasHandle, rng);
            computeModelCorrelations(visible, hidden, d_modelCorrelations, cublasHandle);
            computeDataCorrelations(d_dataCorrelations, d_W, container, cublasHandle, rng);

            weightMatrixUpdate<<<blocks, threads>>>(d_W, d_modelCorrelations,
            	                                d_dataCorrelations, 
            		                        lr, mom, sparsity, N_h, N_v);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        h_spinPtr = h_spinList;
    }

    //Stop timer
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    cudaEventElapsedTime(&time, start, stop);    
    printf("Elapsed time: %f ms\n", time);

    checkCudaErrors(cudaMemcpy(h_W, d_W, sizeof(float)*N_v*N_h, cudaMemcpyDeviceToHost));
    // Save weights 
    FILE *fp_saveW = fopen("W.dat", "w");
    for (int i=0; i < N_v; i++){
        fprintf(fp_saveW, "\n");
        for (int j=0; j < N_h; j++){
            fprintf(fp_saveW, "%f\t", h_W[IDX2F(i,j, N_v)]);
        }
    }
    fclose(fp_saveW);


#ifdef DBUG
 
    copyLayerDeviceToHost(&visible);
    copyLayerDeviceToHost(&hidden);
    checkCudaErrors(cudaMemcpy(h_modelCorrelations, d_modelCorrelations, 
			       sizeof(float)*N_v*N_h, cudaMemcpyDeviceToHost));
    
    FILE *fpW = fopen("dbugW.dat", "w");
    FILE *fph = fopen("dbugHidden.dat", "w");
    FILE *fpv = fopen("dbugVisible.dat", "w");
    
    printf("first spin configuration:\n");
    for (int i=0; i < N_v; i++){
	if (i % N_v == 0)
	    fprintf(fpv, "\n");
        fprintf(fpv, "%f\t", h_spinList[i]);
    }
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

#endif
   
    // Clean up 
    checkCudaErrors(cublasDestroy(cublasHandle));
    checkCudaErrors(curandDestroyGenerator(rng));

    freeLayer(visible); freeLayer(hidden);
    freeCorrContainer(container); 
    freeMemory(&h_W, &d_W, 
	       &h_modelCorrelations, &d_modelCorrelations,
	       &h_dataCorrelations, &d_dataCorrelations,
	       &d_random);

    return EXIT_SUCCESS;
}

__global__
void weightMatrixUpdate(float *d_W, float *d_modelCorrelations,
		        float *d_dataCorrelations, 
			float lr, float mom, float sparsity,
			int N_h, int N_v){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N_h * N_v){
        return;
    }
    float dw = d_W[tid];
    float wUpdate = (d_dataCorrelations[tid] - d_modelCorrelations[tid]);
    d_W[tid] = mom * dw + lr * (1 - mom) * wUpdate - sparsity * fabs(dw);
}


