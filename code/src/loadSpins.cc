#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

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
        int n=0, Nbytes=-1;
       
        while(!feof(fp)){
            ch = fgetc(fp);
    	Nbytes++;
        };
    
#ifdef DBUG
        printf("Loaded %d bytes\n", Nbytes);
#endif 

        float *spins = (float *)malloc(Nbytes * 8 * sizeof(int));
        rewind(fp);
        *Nbits = Nbytes * 8;//Total sample size

        for (int t=0; t < Nbytes; t++){
            ch = fgetc(fp);
            for (int i=0; i < 8; i++){
	        int bit = (ch & (1 << (7-i))) >> (7-i);
                spins[8*t + i] = 2 * bit - 1; 
            }
        }
        fclose(fp);
        return spins;
    }
}
