#ifndef GATING_NETWORK_H
#define GATING_NETWORK_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include "router.h"

extern "C" void gating_network(
    __half* W_gate,
    __half* a,
    __half* logits,
    __half* scores,
    __half* topk_scores,
    __half* topk_indices,
    int batch_size,
    int seq_len,
    int d_hidden,
    int n_experts,
    int topk,
    hipStream_t stream,
    rocblas_handle handle
);


#endif
