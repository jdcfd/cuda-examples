CC:=nvc
CXX:=nvc++
CUC:=nvcc

OPT:= -O3 

CFLAGS:= $(OPT) -fPIE
CUFLAGS:= $(OPT) --use_fast_math -lineinfo -arch=compute_86 -I../Common
CXXFLAGS:= $(OPT)

EXE:=spmv

SRC_CC:=mmio.c
OBJ_CC:=$(patsubst %.c, %.o, $(SRC_CC))

SRC_CU:=matrix_csr.cu mmio_reader.cu vector_dense.cu main.cu 
OBJ_CU:=$(patsubst %.cu, %.o, $(SRC_CU))

ALL_OBJ:=$(OBJ_CC) $(OBJ_CU)

DATA="data/dielFilterV3real.mtx"

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(CUC) $(CUFLAGS) -c $< -o $@

$(EXE): $(ALL_OBJ)
	$(CXX) $(CXXFLAGS) -cuda -gpu=cc86,cuda11.8,lineinfo -cudalib=cusparse -o $(EXE) $(ALL_OBJ)

all: $(EXE)

run: all
	./$(EXE) $(DATA)

debug: all
	cuda-gdb --args ./$(EXE) $(DATA)

prof: all
	nsys profile -t cuda,cublas,nvtx -s none --stats true ./$(EXE) $(DATA)

ncu-shared: all
	ncu --set full --kernel-name sparse_mvm_shared --launch-skip 3 --launch-count 1 -f -o shared-32 "./$(EXE)" $(DATA)

ncu: all
	ncu --set full --kernel-name sparse_mvm --launch-skip 3 --launch-count 1 -f -o noshared-32 "./$(EXE)" $(DATA)

clean:
	rm -f *.o $(EXE)

.PHONY: run debug prof clean

