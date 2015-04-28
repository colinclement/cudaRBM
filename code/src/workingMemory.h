#ifndef __WORKINGMEMORY_H__
#define __WORKINGMEMORY_H__

void allocateMemory(float **d_previousWstep, float **d_random, int N_v, int N_h);

void freeMemory(float *d_previousWstep, float *d_random);

#endif
