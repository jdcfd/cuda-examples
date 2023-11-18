
NVHPC:=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1
FC:=$(NVHPC)/compilers/bin/nvfortran
# INCLUDE:=$(NVHPC)/compilers/include
FFLAGS:=-acc -gpu=cc86 -Minfo=accel -cuda -cudalib=cusparse
EXE:= spmv

SRC_FILES := mmio.f matrix_utils.f90 spmv_driver.F90 
# Create a list of object files (*.o) from source files
OBJ_FILES := $(patsubst %.f, %.o, $(filter %.f, $(SRC_FILES))) \
             $(patsubst %.f90, %.o, $(filter %.f90, $(SRC_FILES))) 
            #  $(patsubst %.F90, %.o, $(filter %.F90, $(SRC_FILES)))

# Pattern rule to compile object files from Fortran source files
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

%.o: %.f
	$(FC) $(FFLAGS) -c $< -o $@

all: program

driver: $(OBJ_FILES)

program: driver
	$(FC) $(FFLAGS) -o $(EXE) $(OBJ_FILES)

clean:
	rm -f *.o *.mod $(EXE)
