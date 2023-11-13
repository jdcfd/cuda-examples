
NVHPC:=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1
FC:=$(NVHPC)/compilers/bin/nvfortran
# INCLUDE:=$(NVHPC)/compilers/include


SRC_DIR := ..
SRC_FILES := $(addprefix $(SRC_DIR), mmio.f matrix_utils.f90 cusparse_routines.f90)
# Create a list of object files (*.o) from source files
OBJ_FILES := $(patsubst %.f, %.o, $(filter %.f, $(SRC_FILES))) \
             $(patsubst %.f90, %.o, $(filter %.f90, $(SRC_FILES))) 

TESTS := test_cusparse test_readmm
TEST_FILES := $(addsuffix .f90, $(TESTS))
TEST_OBJ := $(patsubst %.f90, %.o, $(TEST_FILES)) 

FFLAGS:=-acc -gpu=cc86 -Minfo=accel -cuda -cudalib=cusparse -I$(SRC_DIR)/

# Pattern rule to compile object files from Fortran source files
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

%.o: %.f
	$(FC) $(FFLAGS) -c $< -o $@

all: $(TESTS)

run: $(TESTS)
	$(foreach test,$(TESTS),./$(test);)

test_cusparse: $(SRC_DIR)/cusparse_routines.o test_cusparse.o 
	$(FC) $(FFLAGS) -o test_cusparse test_cusparse.o $(SRC_DIR)/cusparse_routines.o

test_readmm: $(SRC_DIR)/matrix_utils.o $(SRC_DIR)/mmio.o test_readmm.o 
	$(FC) $(FFLAGS) -o test_readmm test_readmm.o $(SRC_DIR)/matrix_utils.o $(SRC_DIR)/mmio.o

clean:
	rm -f *.o *.mod $(EXE)

.PHONY: clean all run