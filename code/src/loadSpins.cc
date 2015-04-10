#include <stdio.h>
#include <stdlib.h>

int* loadSpins(const char *filename) {
    
    FILE *fp = fopen(filename, "r");
    int ch = 0;
    int n=0, Nbytes=-1;
   
    while(!feof(fp)){
        ch = fgetc(fp);
	Nbytes++;
    };
    
    int *spins = (int *)malloc(Nbytes * 8 * sizeof(int));
    rewind(fp);

    int x=0;
    for (int t=0; t < Nbytes; t++){
	ch = fgetc(fp);
	for (int i=0; i < 8; i++){
            spins[8*t + i] = (ch & (1 << (7-i))) >> (7-i);
	}
    }
    fclose(fp);
    return spins;
}
