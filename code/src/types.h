#ifndef __TYPES_H__
#define __TYPES_H__

typedef struct {
    int N_units;
    
    float *h_samples, *d_samples;
    // Store all K samples
    float *d_samplePtr; 
    //Move pointer along during sampling
    int BYTES;
    int SAMPLEBYTES;
    //Size of one sample and all samples
    int kSamples; //k
    //Number of K samples
    
    float *h_conditionalP, *d_conditionalP;
    //Conditional probabilities
    float *h_energySum, *d_energySum;
    //Partial energy
} Layer;

typedef struct {
    int N_v, N_h, batchSize;
    int BATCHBYTES;

    float *d_hiddenRandom;
    float *d_hiddenGivenData;
    float *d_hiddenEnergy;
    float *d_visibleBatch;
} DataCorrContainer;

#endif
