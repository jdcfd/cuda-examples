CC:=nvc
CXX:=nvc++
CUC:=nvcc

CFLAGS:= -fPIE
CUFLAGS:= -O3 -arch=compute_86

EXE:=spmv

SRC_CC:=mmio.c
OBJ_CC:=$(patsubst %.c, %.o, $(SRC_CC))

SRC_CU:=matrix_csr.cu mmio_reader.cu main.cu 
OBJ_CU:=$(patsubst %.cu, %.o, $(SRC_CU))

ALL_OBJ:=$(OBJ_CC) $(OBJ_CU)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(CUC) $(CUFLAGS) -c $< -o $@

$(EXE): $(ALL_OBJ)
	$(CUC) -o $(EXE) $(ALL_OBJ)

all: $(EXE)

run: all
	./$(EXE) "data/cage3.mtx" 4

clean:
	rm *.o $(EXE)

