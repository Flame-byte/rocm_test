#ifndef ROPE_H
#define ROPE_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
void rope(
    __half* d_data,
    const __half* d_cis,
    int seq_len,
    int head_dim,
    int num_heads
);

#endif // ROPE_H