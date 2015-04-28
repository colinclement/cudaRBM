#ifndef __ENERGYFUNCTIONS_H__
#define __ENERGYFUNCTIONS_H__

#include "types.h"

__host__
void allToAll(Layer sampleLayer, Layer givenLayer,
              //const float *d_W, cudaStream_t stream,
              Connection conn, cudaStream_t stream,
              cublasHandle_t handle);

__host__
void convolution(Layer sampleLayer, Layer givenLayer,
                   //const float *d_W, cudaStream_t stream,
                   Connection conn, cudaStream_t stream,
                   cublasHandle_t handle);

__global__
void partialEnergyConvolution(Layer givenLayer, Connection conn);

#endif
