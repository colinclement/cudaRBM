NVCC=nvcc
GCC=g++
EXE=curbm
SRC=main.cu loadSpins.cc
FLAGS=-O3 
#Location of helper_cuda.h
INC=-I/Developer/NVIDIA/CUDA-7.0/samples/common/inc/ -I. 
LIBS=-lcurand -lcublas

GPUOBJS=main.o layers.o sampleCorrelate.o
OBJS=loadSpins.o

default: all

all: $(OBJS) $(GPUOBJS)
	$(NVCC) $(FLAGS) -o $(EXE) $(OBJS) $(GPUOBJS) $(INC) $(LIBS)

loadSpins.o: loadSpins.cc
	$(GCC) $(FLAGS) -c loadSpins.cc

layers.o: layers.cu
	$(NVCC) $(FLAGS) -c layers.cu $(INC)

sampleCorrelate.o: sampleCorrelate.cu
	$(NVCC) $(FLAGS) -c sampleCorrelate.cu $(INC) $(LIBS)

main.o: main.cu
	$(NVCC) $(FLAGS) -c main.cu $(INC) $(LIBS)

.PHONY: clean
clean:
	rm -rf $(EXE) *.o

