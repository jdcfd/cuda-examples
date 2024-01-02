# Sparse Matrix - Dense Vector Multiplication

This is a simple implementation of Sparse matrix-dense vector implementation inspired by this [blog post](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-spmv-docs-spmv_part1/) from AMD. 

Compared to the cusparse Spmv, it is MUCH slower. The main problem is the uncoalesced memory access to the dense vector, which leads to long scoreboard stalls. According to Nsight Compute, this algorithm leads to only 49\% L1TEX cache hit rate! 
