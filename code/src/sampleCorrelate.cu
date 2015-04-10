#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <helper_cuda.h>

#include "types.h"

//Note this is really sig(-x)
#define sig(x) (1.f/(1.f + expf(x)))
#define THREADS_PER 64
#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))


__global__
void _computeAndSample_P(Layer unitLayer, const float *d_random,
		         const int N_units){
    /*   samples conditional probability of visible (hidden) units
     *		d_unit_sample_out : samples of visible (hidden) units
     *		d_P_unit_given_out : P(visible (hidden)=1 | hidden (visible))
     *		d_random : uniform (0,1] random numbers
     *		d_E : 2*W.dot(hidden (visible) samples)
     *		N_units : number of visible (hidden) units
     * */
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= N_units){
        return;
    }
    float P_unit_is_1 = sig(unitLayer.d_energySum[tid]);
    unitLayer.d_conditionalP[tid] = P_unit_is_1;
    unitLayer.d_sample[tid] = 2.f*((float)(P_unit_is_1 > d_random[tid]))-1.f;
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
    float a = -2.f, beta = 0.f;//minus in a instead of in sigmoid
    int N_v = visible.N_units, N_h = hidden.N_units;    
    float *d_visibleRandom = d_random; //just access first N_v elements
    float *d_hiddenRandom = d_random + N_v; //last N_h elements
    dim3 blocks(ceilf((float) MAX(N_v, N_h) / (float) THREADS_PER), 1, 1);
    dim3 threads(THREADS_PER, 1, 1); 
    checkCudaErrors(curandGenerateUniform(rng, d_random, N_v+N_h));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_T, N_v, N_h, &a, d_W, N_v, 
	          	   	d_visibleInitial, 1, &beta, hidden.d_energySum, 1));
    checkCudaErrors(cudaDeviceSynchronize());
    _computeAndSample_P<<<blocks, threads>>>(hidden, d_hiddenRandom, N_h);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, N_v, N_h, &a, d_W, N_v, 
			        hidden.d_sample, 1, &beta, visible.d_energySum, 1));
    checkCudaErrors(cudaDeviceSynchronize());
    _computeAndSample_P<<<blocks, threads>>>(visible, d_visibleRandom, N_v);
    checkCudaErrors(cudaDeviceSynchronize());
}
/*
__host__
void computeK_Gibbs(float *d_k
		    Layer visible, Layer hidden,
		    const float *d_W, float *d_visibleInitial,
		    float *d_random, cublasHandle_t handle,
		    curandGenerator_t, rng){
    
    computeGibbsSample_vhv(visible, hidden, d_W, d_vtest, d_random, handle, rng);
    

}
*/

