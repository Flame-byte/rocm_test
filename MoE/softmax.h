#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" void softmax(
    __half* a,
    __half* b,
    int w,
    int h,
    hipStream_t stream
);

#endif
