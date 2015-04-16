#ifndef __SAMPLECORRELATE_H__
#define __SAMPLECORRELATE_H__

#include "types.h"

__global__
void sampleConditional(Layer unitLayer, const int N_units);
 
 __host__
void computeGibbsSample(Layer sampleLayer, Layer givenLayer,
                        const float *d_W, cudaStream_t stream, 
                        cublasHandle_t handle);

__host__
void computeKGibbs(Layer visible, Layer hidden,
		   const float *d_W, float *d_random,
                   cublasHandle_t handle, curandGenerator_t rng);

__host__
void computeGibbsGivenData(Layer visible, Layer hidden,
                           float *d_W, 
                           cublasHandle_t handle, curandGenerator_t rng);

__host__
void computeCorrelations(Layer visible, Layer hidden,
	                 float *d_correlations, cublasHandle_t handle);

#endif
