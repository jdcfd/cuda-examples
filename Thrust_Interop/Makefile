# Modify this line to your installation path
NVHPC:=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin
FC:=$(NVHPC)/nvfortran
CC:=$(NVHPC)/nvcc
# INCLUDE:=$(NVHPC)/compilers/include
FFLAGS:=-acc -gpu=cc86 -Minfo=accel -cuda -cudalib=cusparse
CUFLAGS:= -arch=compute_86
EXE:= thrust_interop

# No need to add include path to Thrust when using NVHPC compilers

# Pattern rule to compile object files from Fortran source files
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

%.o: %.cu
	$(CC) $(CUFLAGS) -c $< -o $@

%.o: %.cuf
	$(FC) -gpu=cc86 -c $< -o $@

all: $(EXE)
	@echo "Compilation Successfull!"

run: $(EXE)
	./$(EXE)

$(EXE): csort.o thrust_module.o thrust_interop.o
	$(FC) $(FFLAGS) -o $(EXE) thrust_interop.o csort.o thrust_module.o -lstdc++
	
clean:
	rm -f *.o *.mod $(EXE)

.PHONY: clean all run
