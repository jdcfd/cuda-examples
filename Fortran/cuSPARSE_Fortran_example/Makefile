
NVHPC:=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1
FC:=$(NVHPC)/compilers/bin/nvfortran

# INCLUDE:=$(NVHPC)/compilers/include
FFLAGS:=-acc -gpu=cc86 -Minfo=accel -cuda -cudalib=cusparse

EXE:= test_cusparse

SRC_FILES := cusparse_routines.f90 test_cusparse.f90 
# Create a list of object files (*.o) from source files
OBJ_FILES := $(patsubst %.f90, %.o, $(SRC_FILES)) 

# Pattern rule to compile object files from Fortran source files
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

all: $(EXE)

run: $(EXE)
	./$(EXE)

$(EXE): $(OBJ_FILES)
	$(FC) $(FFLAGS) -o $(EXE) $(OBJ_FILES)	

clean:
	rm -f *.o *.mod $(EXE)

.PHONY: clean run all
