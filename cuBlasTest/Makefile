LOAD	= module purge && module load cuda/5.5

DEFAULT	= cuBlasTest
OBJECTS	= $(DEFAULT).o
LSF		= $(DEFAULT).lsf

CUDAPATH=/lustre/utility/cuda/5.5
CUDALIB=$(CUDAPATH)/lib64
CUDAINC=$(CUDAPATH)/include
#CUDA_LIBCALL= -lcudart -lcuda
CUDA_LIBCALL= -lcublas

CC		= $(LOAD) && nvcc
RUN		= bsub -q gpu < $(LSF)

all: $(DEFAULT)

%.o : %.c
	$(CC) $(CFLAGS) -c $<

%.o : %.cu
	$(CC) $(CFLAGS) -c $<

$(DEFAULT) : $(DEFAULT).o
	$(CC) $(CFLAGS) -o $@ $^ -L$(CUDALIB) -I$(CUDAINC) $(CUDA_LIBCALL)
	-@rm -f *.o 2>&1 > /dev/null

run : 
	$(RUN)


.PHONY: clean clear all

clean :
	-@rm -f *.o $(DEFAULT) 2>&1 > /dev/null

clear :
	-@rm -f *.o $(DEFAULT) [0-9]*.err [0-9]*.out 2>&1 > /dev/null
