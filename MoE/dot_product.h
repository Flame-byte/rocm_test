#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>

extern "C" void dot_product(
    const __half* a,
    const __half* b,
    __half* c,
    const int n,
    hipStream_t stream
);

extern "C" void dot_product_scores(
    const __half* a,    
    const __half* b,
    const __half* scores,
    __half* c,
    const int seq_len,
    const int d_expert,
    hipStream_t stream
);

#endif  // DOT_PRODUCT_H
