#ifndef FLASH_ENCODING_H
#define FLASH_ENCODING_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>


void flash_encoding(const __half* Q,
                     const __half* K,
                     const __half* V,
                     __half* O,
                     const int seq_len_q,
                     const int seq_len_kv,
                     const int head_dim,
                     const int num_heads_q,
                     const int num_heads_kv);

#endif // FLASH_ENCODING_H