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
              Connection conn, cudaStream_t stream,
              cublasHandle_t handle){
    int N_v = conn.fan_in, N_h = conn.fan_out;
    float a = 1.f, beta = 0.f;
    cublasOperation_t OP; 
    OP = ((givenLayer.N_units == conn.cols) ? CUBLAS_OP_N : CUBLAS_OP_T);
    checkCudaErrors(cublasSgemv(handle, OP, N_v, N_h, &a, conn.d_W, N_v, 
	          	   	givenLayer.d_samplePtr, 1, &beta, 
                    sampleLayer.d_energySum, 1));
}

__host__
void convolution(Layer sampleLayer, Layer givenLayer,
                 Connection conn, cudaStream_t stream,
                 cublasHandle_t handle){
    return;
}

__global__
void partialEnergyConvolution(Layer givenLayer, Connection conn){
    return;
}

