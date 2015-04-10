#ifndef __TYPES_H__
#define __TYPES_H__

typedef struct {
    float *h_sample, *d_sample;
    float *h_conditionalP, *d_conditionalP;
    float *h_energySum, *d_energySum;
    int N_units;
    int BYTES;
} Layer;

#endif
