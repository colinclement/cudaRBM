#ifndef __LAYERS_H__
#define __LAYERS_H__

#include "types.h"

__host__
void copyLayerDeviceToHost(Layer *unitLater);

__host__
void allocateLayer(Layer *newLayer, int N_units, int kSamples);

__host__
void freeLayer(Layer newLayer);

__host__
void allocateCorrContainer(DataCorrContainer *container, 
		           int N_v, int N_h, int batchSize);
__host__
void freeCorrContainer(DataCorrContainer container);

#endif
