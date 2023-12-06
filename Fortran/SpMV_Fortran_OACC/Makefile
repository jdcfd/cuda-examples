
NVHPC:=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1
FC:=$(NVHPC)/compilers/bin/nvfortran
CC:=$(NVHPC)/compilers/bin/nvcc
# INCLUDE:=$(NVHPC)/compilers/include
FFLAGS:= -g -acc -gpu=cc86 -Minfo=accel -cuda -cudalib=cusparse
CUFLAGS:= -g -arch=compute_86
CUFFLAGS:= -g -gpu=cc86
EXE:= spmv

SRC_FILES := mmio.f matrix_utils.f90 main.f90 
# Create a list of object files (*.o) from source files
OBJ_FILES := $(patsubst %.f, %.o, $(filter %.f, $(SRC_FILES))) \
             $(patsubst %.f90, %.o, $(filter %.f90, $(SRC_FILES))) 
            #  $(patsubst %.F90, %.o, $(filter %.F90, $(SRC_FILES)))

CU_FILES := csort.cu
CU_OBJS := $(patsubst %.cu, %.o, $(CU_FILES))
CUF_FILES := thrust_module.cuf 
CUF_OBJS := $(patsubst %.cuf, %.o, $(CUF_FILES))

# Pattern rule to compile object files from Fortran source files
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

%.o: %.f
	$(FC) $(FFLAGS) -c $< -o $@

%.o: %.cu
	$(CC) $(CUFLAGS) -c $< -o $@

%.o: %.cuf
	$(FC) $(CUFFLAGS) -c $< -o $@

all: $(EXE)

run: all
	./$(EXE)

$(EXE): $(CU_OBJS) $(CUF_OBJS) $(OBJ_FILES)
	$(FC) $(FFLAGS) -o $(EXE) $(CU_OBJS) $(CUF_OBJS) $(OBJ_FILES) -lstdc++

clean:
	rm -f *.o *.mod $(EXE)

.PHONY: clean all run
