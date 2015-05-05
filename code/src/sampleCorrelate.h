#ifndef __SAMPLECORRELATE_H__
#define __SAMPLECORRELATE_H__

#include "types.h"

__global__
void sampleConditional(Layer unitLayer, const int N_units);
 
 __host__
void computeGibbsSample(Layer sampleLayer, Layer givenLayer,
                        //const float *d_W, energyFunc energy,
                        Connection conn, energyFunc energy,
                        cudaStream_t stream, cublasHandle_t handle); 
                        //cublasHandle_t handle);

__host__
void computeKGibbs(Layer visible, Layer hidden,
		           //const float *d_W, energyFunc energy,
                   Connection conn, energyFunc energy,
                   float *d_random, curandGenerator_t rng,
                   cudaStream_t stream, cublasHandle_t handle);

__host__
void computeGibbsGivenData(Layer visible, Layer hidden,
                           //float *d_W, energyFunc energy,
                           Connection conn, energyFunc energy,
                           curandGenerator_t rng,
                           cudaStream_t stream, cublasHandle_t handle);

__host__
void computeCorrelations(Layer visible, Layer hidden,
                         float *d_correlations, cublasHandle_t handle);

#endif
