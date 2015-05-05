NVCC=nvcc
GCC=g++
EXE=curbm
SRC=main.cu loadSpins.cc
FLAGS=-O3 -use_fast_math 
LIBS=-lcurand -lcublas

#Location of helper_cuda.h
HOST=$(shell hostname)
ifeq ($(HOST), spicable.lassp.cornell.edu)
INC=-I/Developer/NVIDIA/CUDA-7.0/samples/common/inc -I. 
endif
ifeq ($(HOST), dain)
INC=-I/opt/cuda5.5/samples/common/inc -I.
endif
ifeq ($(HOST), jection)
INC= -I$(CUDA_HOME)/samples/common/inc -I.
endif

GPUOBJS=main.o layers.o sampleCorrelate.o workingMemory.o energyFunctions.o
OBJS=loadSpins.o

default: main

main: $(OBJS) $(GPUOBJS)
	$(NVCC) $(FLAGS) -o $(EXE) $(OBJS) $(GPUOBJS) $(INC) $(LIBS)

loadSpins.o: loadSpins.cu
	$(NVCC) $(FLAGS) -c loadSpins.cu $(INC)

layers.o: layers.cu
	$(NVCC) $(FLAGS) -c layers.cu $(INC)

sampleCorrelate.o: sampleCorrelate.cu
	$(NVCC) $(FLAGS) -c sampleCorrelate.cu $(INC) $(LIBS)

workingMemory.o: workingMemory.cu
	$(NVCC) $(FLAGS) -c workingMemory.cu $(INC)

energyFunctions.o: energyFunctions.cu
	$(NVCC) $(FLAGS) -c energyFunctions.cu $(INC)

main.o: main.cu
	$(NVCC) $(FLAGS) -c main.cu $(INC) $(LIBS)

.PHONY: clean
clean:
	rm -rf $(EXE) *.o

