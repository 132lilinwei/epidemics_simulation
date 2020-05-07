DEBUG=0
CC=gcc
OMP=-fopenmp -DOMP
CFLAGS=-g -O3 -Wall -DDEBUG=$(DEBUG)
#CFLAGS=-g -O3 -Wall -DDEBUG=$(DEBUG) -DDYNAMIC
LDFLAGS= -lm -L/usr/local/depot/cuda-10.2/lib64/ -lcudart
DDIR = ./data

CFILES = crun.c graph.c simutil.c sim.c rutil.c cycletimer.c instrument.c
OFILES = crun.o graph.o simutil.o sim.o rutil.o cycletimer.o instrument.o
HFILES = crun.h rutil.h cycletimer.h

CXX=g++ -m64
CXXFLAGS=-O3 -Wall

NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61

all: crun-gpu

crun-gpu: $(CFILES) $(HFILES) $(OFILES) sim_cuda.o fake_omp.o
	$(CXX) $(CXXFLAGS) -o $@ sim_cuda.o fake_omp.o $(OFILES) $(LDFLAGS)


$(OFILES): $(CFILES) $(HFILES)
	$(CC)  $(CFLAGS) -c $(CFILES)

fake_omp.o: fake_omp.c fake_omp.h
	$(CC) $(CFLAGS) -c fake_omp.c

sim_cuda.o: sim_cuda.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

clean:
	rm -f *~ *.pyc
	rm -rf *.dSYM
	rm -rf regression-cache check
	rm -f crun crun-seq crun-omp
	rm -f *.o
