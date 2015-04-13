#ifndef __WORKINGMEMORY_H__
#define __WORKINGMEMORY_H__

void allocateMemory(float **h_W, float **d_W, float **d_previousWstep, 
		    float **h_modelCorrelations, float **d_modelCorrelations,
		    float **h_dataCorrelations, float **d_dataCorrelations,
		    float **d_random, int N_v, int N_h);

void freeMemory(float **h_W, float **d_W, float **d_previousWstep,
		float **h_modelCorrelations, float **d_modelCorrelations,
		float **h_dataCorrelations, float **d_dataCorrelations,
		float **d_random);

#endif
