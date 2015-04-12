#ifndef __WORKINGMEMORY_H__
#define __WORKINGMEMORY_H__

void allocateMemory(float **h_W, float **d_W, 
		    float **h_modelCorrelations, float **d_modelCorrelations,
		    float **h_dataCorrelations, float **d_dataCorrelations,
		    float **d_random, float **d_hiddenRandom,
		    float **d_hiddenGivenData, float **d_hiddenEnergy,
		    int N_v, int N_h);

void freeMemory(float **h_W, float **d_W,
		float **h_modelCorrelations, float **d_modelCorrelations,
		float **h_dataCorrelations, float **d_dataCorrelations,
		float **d_random, float **d_hiddenRandom,
		float **d_hiddenGivenData, float **d_hiddenEnergy);

#endif
