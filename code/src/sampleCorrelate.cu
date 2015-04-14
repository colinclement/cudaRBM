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

/*
#ifdef DBUG_K
    float *h_visibleTest = (float *)malloc(visible.BYTES);
    checkCudaErrors(cudaMemcpy(h_visibleTest, d_tempPtr, visible.BYTES,
			       cudaMemcpyDeviceToHost));
    for (int i=0; i < N_v; i++){
	if (i % 6 == 0){
	    printf("\n");
	}
        printf("%f\t", h_visibleTest[i]);
    }; printf("\n");
    free(h_visibleTest); h_visibleTest = NULL;
#endif
*/

__global__
void _computeAndSample_P(Layer unitLayer, const float *d_random, const int N_units){
    /*   samples conditional probability of visible (hidden) units
     *          unitLayer : an instance of Layer (hidden or visible)
     *		d_random : uniform (0,1] random numbers
     * */
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_units){
        return;
    }
    float P_unit_is_1 = sig(unitLayer.d_energySum[tid]);
    unitLayer.d_conditionalP[tid] = P_unit_is_1;
    unitLayer.d_samplePtr[tid] = 2.f*((float)(P_unit_is_1 > d_random[tid]))-1.f;
}

__host__
void computeGibbsSample_vhv(Layer visible, Layer hidden,
		            const float *d_W, float *d_visibleInitial,
			    float *d_random, cublasHandle_t handle, 
			    curandGenerator_t rng){
     /*    Computes energies, samples conditional probabilities by Gibbs sampling.
     *        d_visible(hidden)Sample : Sample of visible (hidden) units
     *        d_visible(hidden)CondP : Conditional probability
     *        d_W : weight matrix of size N_v by N_h
     *        d_visibleInitial : Sample of visible units for starting Markov chain
     *        d_visible(hidden)Energies : energy partial sums for computing probabilities 
     *        		i.e. for visible = 2 sum_k W_ik h_k for some hidden sample
     *        d_random : array for storing randoom numbers (N_v + N_h of them)
     *        N_v and N_h : numbers of visible and hidden units
     *	      handle, rng : cuBLAS handle and cuRAND random number generator 
     * */
    float a = -2.f, beta = 0.f;//minus in E instead of in sigmoid
    int N_v = visible.N_units, N_h = hidden.N_units;    
    float *d_visibleRandom = d_random; //just access first N_v elements
    float *d_hiddenRandom = d_random + N_v; //last N_h elements
    dim3 hblocks(ceilf((float) N_h / (float) THREADS_PER), 1, 1);
    dim3 vblocks(ceilf((float) N_v / (float) THREADS_PER), 1, 1);
    dim3 threads(THREADS_PER, 1, 1); 
    cudaStream_t stream; checkCudaErrors(cublasGetStream(handle, &stream)); 
    checkCudaErrors(curandGenerateUniform(rng, d_random, N_v+N_h));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_T, N_v, N_h, &a, d_W, N_v, 
	          	   	d_visibleInitial, 1, &beta, hidden.d_energySum, 1));
    checkCudaErrors(cudaDeviceSynchronize());
    
    _computeAndSample_P<<<hblocks, threads, 0, stream>>>(hidden, d_hiddenRandom, N_h);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, N_v, N_h, &a, d_W, N_v, 
			        hidden.d_samplePtr, 1, &beta, visible.d_energySum, 1));
    checkCudaErrors(cudaDeviceSynchronize());
    
    _computeAndSample_P<<<vblocks, threads, 0, stream>>>(visible, d_visibleRandom, N_v);
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__
void computeK_Gibbs(Layer visible, Layer hidden,
		    const float *d_W, float *d_visibleInitial,
		    float *d_random, cublasHandle_t handle,
		    curandGenerator_t rng){
    int N_v = visible.N_units, N_h = hidden.N_units;    
    float *d_tempPtr = d_visibleInitial; 
    int k = visible.kSamples;

    for (int i=0; i < k; i++){
        computeGibbsSample_vhv(visible, hidden, d_W, d_tempPtr, d_random, handle, rng);
        d_tempPtr = visible.d_samplePtr; //Previous sample
        visible.d_samplePtr += N_v; //Advance pointers
        hidden.d_samplePtr += N_h;
    } 
    visible.d_samplePtr = visible.d_samples;//Reset moving pointer
    hidden.d_samplePtr = hidden.d_samples;
}

__host__
void computeModelCorrelations(Layer visible, Layer hidden,
		              float *d_modelCorrelations, cublasHandle_t handle){
    int k = visible.kSamples, N_v = visible.N_units, N_h = hidden.N_units;
    //float *d_visiblePtr = visible.d_samples, *d_hiddenPtr = hidden.d_samples;
    const float alpha = 1.f/((float) k), beta = 0.f;
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
			        N_v, N_h, k, &alpha, visible.d_samples, N_v, 
				hidden.d_samples, N_h, &beta, d_modelCorrelations, N_v));
}

__global__
void _sampleH_GivenData(DataCorrContainer container, const int N_units){
    /*   samples conditional probability of hidden units
     *          d_energySum : partial energy sum for hidden layers
     *          d_random : uniform (0,1] random numbers
     *          N_units : Number of hidden units
     * */
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_units){
        return;
    }
    float P_unit_is_1 = sig(container.d_hiddenEnergy[tid]);
    float rnd = container.d_hiddenRandom[tid];
    container.d_hiddenGivenData[tid] = 2.f*((float)(P_unit_is_1 > rnd))-1.f;
}

__host__
void computeDataCorrelations(float *d_dataCorrelations, 
		             float *d_W, DataCorrContainer container, 
			     cublasHandle_t handle, curandGenerator_t rng){
    float *d_tempPtr = container.d_visibleBatch;
    int N_v = container.N_v, N_h = container.N_h, batchSize = container.batchSize;
    dim3 blocks(ceilf((float) N_h / (float) THREADS_PER), 1, 1);
    dim3 threads(THREADS_PER, 1, 1);
    cudaStream_t stream; checkCudaErrors(cublasGetStream(handle, &stream)); 
    float a = -2.f, beta = 0.f, alpha = 1.f/((float)batchSize); //minus in E instead of in sigmoid
    checkCudaErrors(cudaMemsetAsync(d_dataCorrelations, 0, N_v*N_h*sizeof(float), stream));
    for (int i = 0; i < batchSize; i++){
        checkCudaErrors(curandGenerateUniform(rng, container.d_hiddenRandom, N_h));
        checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_T, N_v, N_h, &a, d_W, N_v, 
	              	   	    d_tempPtr, 1, &beta, container.d_hiddenEnergy, 1));
	_sampleH_GivenData<<<blocks, threads, 0, stream>>>(container, N_h);
        checkCudaErrors(cublasSger(handle, N_v, N_h, &alpha, d_tempPtr, 1, container.d_hiddenGivenData, 1,
                                   d_dataCorrelations, N_v));
        d_tempPtr += N_v;
    }
}

// S_v = (s_i x n) visible samples
// S_h = (s_k x n) hidden samples
// S_v.dot(S_h.T) = (s_i x s_k) matrix
//General matrix multiplication:
//cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
//                           int m, int n, int k, const float *alpha, const float *A, int lda, 
//			     const float *B, int ldb, const float *beta, float *C, int ldc)
//
//NOTES:
//Rank-1 update for v_i h_k correlation matrix
//cublasSger(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, 
//           int incx, const float *y, int incy, float *A, int lda)
//TODO: Consider updating weight matrix in place (probabably a lot more efficient).
//      Also consider concurrent updating and sampling.

