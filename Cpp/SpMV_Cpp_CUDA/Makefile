CC:=nvc
CXX:=nvc++
CUC:=nvcc

CFLAGS:= -fPIE
CUFLAGS:= -O3 -arch=compute_86 -I../Common

EXE:=spmv

SRC_CC:=mmio.c
OBJ_CC:=$(patsubst %.c, %.o, $(SRC_CC))

SRC_CU:=matrix_csr.cu mmio_reader.cu vector_dense.cu main.cu 
OBJ_CU:=$(patsubst %.cu, %.o, $(SRC_CU))

ALL_OBJ:=$(OBJ_CC) $(OBJ_CU)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(CUC) $(CUFLAGS) -c $< -o $@

$(EXE): $(ALL_OBJ)
	$(CXX) -O3 -cuda -cudalib=cusparse -o $(EXE) $(ALL_OBJ)

all: $(EXE)

run: all
	./$(EXE) "data/1138_bus.mtx" #> stdout.txt 2>&1

prof: all
	nsys nvprof ./$(EXE) "data/1138_bus.mtx"

clean:
	rm -f *.o $(EXE)

