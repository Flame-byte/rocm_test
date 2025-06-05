#ifndef MOE_INFER_H
#define MOE_INFER_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>

extern "C" void moe_infer(
    const __half* x,           // input tokens: shape (M, d_hidden)
    const __half* topk_scores, // gating scores: shape (M, topk)
    const __half* topk_indices,// gating indices: shape (M, topk)
    int batch_size,
    int seq_len,
    int d_hidden,
    int n_experts,
    int topk
);




#endif
