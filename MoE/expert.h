#ifndef EXPERT_H
#define EXPERT_H

#include <hip/hip_runtime.h> 
#include <hip/hip_fp16.h>
#include <iostream>        
#include <vector>          
#include <rocblas/rocblas.h> 

extern "C" void expert_shared(
    __half* W_up,
    __half* W_gate,
    __half* W_down,
    __half* a,  //input(batch_size, seq_len, d_hidden)
    __half* b,  //output(batch_size, seq_len, d_hidden)
    __half* UP_cache,
    __half* GATE_cache,
    __half* DOWN_cache,
    __half* SiLU_cache,
    int batch_size,
    int seq_len,
    int d_hidden,
    int d_expert,
    int n_experts,
    hipStream_t stream,
    rocblas_handle handle
);

extern "C" void expert_router(
    __half* W_up,
    __half* W_gate,
    __half* W_down,
    __half* scores,
    __half* a,  //input(seq_len, d_hidden)
    __half* b,  //output(seq_len, d_hidden)
    __half* UP_cache,
    __half* GATE_cache,
    __half* DOWN_cache,
    __half* SiLU_cache,
    int seq_len,
    int d_hidden,
    int d_expert,
    hipStream_t stream,
    rocblas_handle handle
);

#endif  // EXPERT_H
