#ifndef __TYPES_H__
#define __TYPES_H__

typedef struct {
    int N_units, numSamples, BYTES, SAMPLEBYTES;
    
    float *h_samples, *d_samples, *d_samplePtr;
    
    float *h_random, *d_random; //for sampling 
    float *h_conditionalP, *d_conditionalP;
    float *h_energySum, *d_energySum;
} Layer;

#endif
