#include <stdlib.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#include "types.h"

#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define IDX2F(i,j,ld) (((j)*(ld))+(i))

__host__
void allToAll(Layer sampleLayer, Layer givenLayer,
              const float *d_W, cudaStream_t stream,
              cublasHandle_t handle){
    int sN = sampleLayer.N_units, gN = givenLayer.N_units;
    int N_v = MAX(sN, gN), N_h = MIN(sN, gN);
    float a = 1.f, beta = 0.f;
    cublasOperation_t OP = ((sN > gN) ? CUBLAS_OP_N : CUBLAS_OP_T);
    checkCudaErrors(cublasSgemv(handle, OP, N_v, N_h, &a, d_W, N_v, 
	          	   	givenLayer.d_samplePtr, 1, &beta, 
                    sampleLayer.d_energySum, 1));
}

__host__
void convolution(Layer sampleLayer, Layer givenLayer,
                 const float *d_W, cudaStream_t stream,
                 cublasHandle_t handle){
    return;
}

