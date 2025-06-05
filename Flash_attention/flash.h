#ifndef FLASH_H
#define FLASH_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

void flash_attention(const __half* Q,
                     const __half* K,
                     const __half* V,
                     __half* O,
                     const int seq_len,
                     const int head_dim,
                     const int num_heads_q,
                     const int num_heads_kv);

#endif // FLASH_H