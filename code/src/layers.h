#ifndef __LAYERS_H__
#define __LAYERS_H__

#include "types.h"

__host__
void copyLayerDeviceToHost(Layer unitLater);

__host__
Layer allocateLayer(int N_units, int numSamples);

__host__
void updateLayerSample(Layer unitLayer, float *h_hostSamples,
                       int  BYTES, cudaStream_t stream);

__host__
void freeLayer(Layer newLayer);

#endif
