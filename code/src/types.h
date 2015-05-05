#ifndef __TYPES_H__
#define __TYPES_H__
#include "cuda.h"
#include "cublas_v2.h"

enum Filter {ALLTOALL, CONVOLUTION};

typedef struct {
    int N_units, numSamples, BYTES, SAMPLEBYTES;
    
    float *h_samples, *d_samples, *d_samplePtr;
    
    float *h_random, *d_random; //for sampling 
    float *h_conditionalP, *d_conditionalP;
    float *h_energySum, *d_energySum;
} Layer;

typedef struct {
    int fan_in, fan_out; //N_v, N_h
    Filter filterType;
    int width, stride; //convolution only
    int rows, cols; //h_W, shape
    int FILTERBYTES; //size of h_W
    int CORRBYTES; //Size of h_<>Correlations

    float *h_W, *d_W;
    float *h_modelCorrelations, *d_modelCorrelations;
    float *h_dataCorrelations, *d_dataCorrelations;
} Connection;

typedef void (* energyFunc)(Layer sampleLayer, Layer givenLayer,
                            Connection conn, cudaStream_t stream,
                            cublasHandle_t handle);

#endif
