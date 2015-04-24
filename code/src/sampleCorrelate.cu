#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <helper_cuda.h>

#include "types.h"

//#define DBUG_K
//#define DBUG_GIBBS

//Note this is really sig(-x)
#define sig(x) (1.f/(1.f + expf(x)))
#define THREADS_PER 32
#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define IDX2F(i,j,ld) (((j)*(ld))+(i))

__global__
void sampleConditional(Layer unitLayer, const int N_units){
    /*   samples conditional probability of visible (hidden) units
     *          unitLayer : an instance of Layer (hidden or visible)
     *		d_random : uniform (0,1] random numbers
     * */
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_units){
        return;
    }
    float P_unit_is_1 = sig(-2.f * unitLayer.d_energySum[tid]);
    //unitLayer.d_conditionalP[tid] = P_unit_is_1;
    float rnd = unitLayer.d_random[tid]; 
    unitLayer.d_samplePtr[tid] = 2.f*((float)(P_unit_is_1 > rnd))-1.f;
}

__host__
void computeGibbsSample(Layer sampleLayer, Layer givenLayer,
                        const float *d_W, energyFunc energy,
                        cudaStream_t stream, cublasHandle_t handle){
                        //cudaStream_t stream, cublasHandle_t handle){
    // Sample state of sampleLayer given state of givenLayer
    // NOTE: Assumes visible Layer has MORE units than hidden Layer!!!
    int sN = sampleLayer.N_units;
    //float a = -2.f, beta = 0.f;//minus in E instead of in sigmoid
    //int N_v = MAX(sN, gN), N_h = MIN(sN, gN);
    //cublasOperation_t OP = ((sN > gN) ? CUBLAS_OP_N : CUBLAS_OP_T);
    dim3 blocks((int) ceilf((float) sN / (float) THREADS_PER), 1, 1);
    dim3 threads(THREADS_PER, 1, 1);
    //Can put energy functional here
    //Energy functional should take two layers (sample and given),
    //And should populate sample.d_energySum
    //checkCudaErrors(cublasSgemv(handle, OP, N_v, N_h, &a, d_W, N_v, 
	//          	   	givenLayer.d_samplePtr, 1, &beta, 
    //                sampleLayer.d_energySum, 1));
    energy(sampleLayer, givenLayer, d_W, stream, handle);
    sampleConditional<<<blocks, threads, 0, stream>>>(sampleLayer, sN);
}

__host__
void computeKGibbs(Layer visible, Layer hidden,
                   const float *d_W, energyFunc energy, 
                   float *d_random, curandGenerator_t rng,
                   cudaStream_t stream, cublasHandle_t handle){
    int N_v = visible.N_units, N_h = hidden.N_units;    
    visible.d_samplePtr = visible.d_samples; 
    for (int i=0; i < visible.numSamples; i++){
        checkCudaErrors(curandGenerateUniform(rng, d_random, N_v+N_h));
        visible.d_random = d_random; hidden.d_random = d_random + N_v;
        hidden.d_samplePtr = hidden.d_samples + i * N_h;
        computeGibbsSample(hidden, visible, d_W, energy, stream, handle);
        visible.d_samplePtr = visible.d_samples + i * N_v;
        computeGibbsSample(visible, hidden, d_W, energy, stream, handle);
    } 
    visible.d_samplePtr = visible.d_samples;//Reset moving pointer
    hidden.d_samplePtr = hidden.d_samples;
}

__host__
void computeGibbsGivenData(Layer visible, Layer hidden,
                           float *d_W, energyFunc energy, 
                           curandGenerator_t rng, 
                           cudaStream_t stream, cublasHandle_t handle){
    int N_v = visible.N_units, N_h = hidden.N_units;    
    for (int i = 0; i < visible.numSamples; i++){
        checkCudaErrors(curandGenerateUniform(rng, hidden.d_random, N_h));
        hidden.d_samplePtr = hidden.d_samples + i * N_h;
        visible.d_samplePtr = visible.d_samples + i * N_v;
        computeGibbsSample(hidden, visible, d_W, energy, stream, handle);
    }
}

__host__
void computeCorrelations(Layer visible, Layer hidden,
		                 float *d_correlations, cublasHandle_t handle){
    int k = visible.numSamples, N_v = visible.N_units, N_h = hidden.N_units;
    const float alpha = 1.f/((float) k), beta = 0.f;
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
			        N_v, N_h, k, &alpha, visible.d_samples, N_v, 
				    hidden.d_samples, N_h, &beta, d_correlations, N_v));
}


//TODO: Investigate function pointers for improved modularity in transitioning to CRBM

//TODO: Consider updating weight matrix in place (probabably a lot more efficient).
//      Also consider concurrent updating and sampling.

