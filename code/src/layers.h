#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <cuda.h>
#include <curand.h>
#include "types.h"

__host__
void copyLayerDeviceToHost(Layer unitLater);

__host__
Layer allocateLayer(int N_units, int numSamples);

__host__
Connection allocateConnection(int N_v, int N_h, curandGenerator_t rng,
                              Filter filterType, int width, int stride);
__host__
void updateLayerSample(Layer unitLayer, float *h_hostSamples,
                       int  BYTES, cudaStream_t stream);

__host__
void freeLayer(Layer newLayer);

__host__
void freeConnection(Connection newConn);

#endif
