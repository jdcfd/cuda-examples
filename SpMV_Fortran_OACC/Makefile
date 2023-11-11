
NVHPC:=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1
FC:=$(NVHPC)/compilers/bin/nvfortran
INCLUDE:=$(NVHPC)/compilers/include
FFLAGS:=-acc -gpu=cc86 -Minfo=accel -cudalib=cutensor #-I$(INCLUDE)

# SRC_FILES := $(wildcard *.f90)
SRC_FILES := prec_const.f90 matrix_utils.f90

# Create a list of object files (*.o) from source files
OBJ_FILES := $(patsubst %.f90, %.o, $(SRC_FILES))

# Pattern rule to compile object files from Fortran source files
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

all: program

mmio.o: mmio.f
	$(FC) -c mmio.f

driver: mmio.o $(OBJ_FILES) spmv_driver.F90
	$(FC) $(FFLAGS) -c spmv_driver.F90 

program: driver
	$(FC) $(FFLAGS) -o spmv spmv_driver.o $(OBJ_FILES) mmio.o

clean:
	rm -f *.o spmv
