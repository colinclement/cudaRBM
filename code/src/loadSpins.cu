#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <cuda.h>
#include <helper_cuda.h>

//#define DBUG

float* loadSpins(const char *filename, int *Nbits) {
    
    FILE *fp = fopen(filename, "r");
    int errnum = 0;
    if (fp == NULL){
        errnum = errno;
	fprintf(stderr, "Error opening file: %s\n", strerror(errnum));
	return NULL;
    }
    else{

        int ch = 0;
        int Nbytes=-1;
       
        while(!feof(fp)){
            ch = fgetc(fp);
    	    Nbytes++;
        };
    
#ifdef DBUG
        printf("Loaded %d bytes\n", Nbytes);
#endif 

        //float *spins = (float *)malloc(Nbytes * 8 * sizeof(int));
	float *h_spins;
        cudaError_t status = cudaMallocHost((void **)&h_spins, Nbytes*8*sizeof(float));
	if (status != cudaSuccess){
	    printf("Error allocating pinned host memory\n");
	    h_spins = (float *)malloc(Nbytes * 8 * sizeof(float));
	}	
	rewind(fp);
        *Nbits = Nbytes * 8;//Total sample size

        for (int t=0; t < Nbytes; t++){
            ch = fgetc(fp);
            for (int i=0; i < 8; i++){
	        int bit = (ch & (1 << (7-i))) >> (7-i);
                h_spins[8*t + i] = 2 * bit - 1; 
            }
        }
        fclose(fp);
        return h_spins;
    }
}
