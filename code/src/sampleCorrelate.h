#ifndef __SAMPLECORRELATE_H__
#define __SAMPLECORRELATE_H__

#include "types.h"

__global__ void _computeAndSample_P(Layer unitLayer, const float *d_random,
		         const int N_units);
 
__host__ void computeGibbsSample_vhv(Layer visible, Layer hidden,
		            const float *d_W, float *d_visibleInitial,
			    float *d_random, cublasHandle_t handle, 
			    curandGenerator_t rng);
 
__host__ void computeK_Gibbs(Layer visible, Layer hidden,
		    const float *d_W, float *d_visibleInitial,
		    float *d_random, cublasHandle_t handle,
		    curandGenerator_t rng);

__host__ void computeModelCorrelations(Layer visible, Layer hidden,
		                       float *d_modelCorrelations, 
				       cublasHandle_t handle);

__global__ void _sampleH_GivenData(float *d_hiddenSample, 
		                   const float *d_energySum,
	        	           const float *d_random, const int N_units);

__host__ void computeDataCorrelations(float *d_dataCorrelations, 
		                      float *d_W, float *d_spinData, float *d_hiddenRandom,
			              float *d_hiddenGivenData, float *d_hiddenEnergy,
		                      const int N_v, const int N_h, const int batchSize, 
			              cublasHandle_t handle, curandGenerator_t rng);

#endif
