#ifndef LINEAR_H
#define LINEAR_H

#include <hip/hip_runtime.h> 
#include <hip/hip_fp16.h>
#include <iostream>        
#include <vector>          
#include <rocblas/rocblas.h> 

extern "C" void linear(
    __half* a,
    __half* b,
    __half* d,
    int batch_size,
    int seq_len,
    int d_hidden,
    int d_expert,
    rocblas_handle handle
);

#endif  // LINEAR_H