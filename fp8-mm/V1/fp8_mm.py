# This script provides a template for using load_inline to run a HIP kernel for
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
CPP_WRAPPER = """
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c);
"""

CUDA_SRC = """
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>

#define TILE_SIZE 32

__global__ void matmul_fp8_to_bf16(
    const __hip_fp8_e4m3_fnuz* a, 
    const __hip_fp8_e4m3_fnuz* b, 
    __hip_bfloat16* c, 
    float* a_scale, 
    float* b_scale, 
    int m, 
    int n, 
    int k) {
    __shared__ __hip_fp8_e4m3_fnuz tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __hip_fp8_e4m3_fnuz tile_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_a = bx * TILE_SIZE + ty;
    int row_b = by * TILE_SIZE + tx;

    float result = 0.0f;
    for (int i = 0; i < k; i += TILE_SIZE){
        int k_start = i;

        int a_col = k_start + tx;
        int a_idx = a_col * m + row_a;
        if (a_col < k && row_a < m)
            tile_A[ty][tx] = a[a_idx];
        else
            tile_A[ty][tx] = static_cast<__hip_fp8_e4m3_fnuz>(0);

        int b_col = k_start + ty;
        int b_idx = b_col * n + row_b;
        if (b_col < k && row_b < n)
            tile_B[ty][tx] = b[b_idx];
        else
            tile_B[ty][tx] = static_cast<__hip_fp8_e4m3_fnuz>(0);

        __syncthreads();
        float block_result = 0;
        for (int j = 0; j < TILE_SIZE; ++j) {
            float av = static_cast<float>(tile_A[ty][j]);
            float bv = static_cast<float>(tile_B[j][tx]);
            block_result += av * bv;
        }
        result += block_result * a_scale[row_a + (i / 128) * m] * b_scale[(row_b / 128) + (i / 128) * ((n + 127) / 128)];
        __syncthreads();
    }

    if (row_a < m && row_b < n) {
        c[row_a * n + row_b] = static_cast<__hip_bfloat16>(result);
    }
}

void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim( (m + TILE_SIZE - 1) / TILE_SIZE ,(n + TILE_SIZE - 1) / TILE_SIZE );

    custom_kernel<<<gridDim, blockDim>>>( (__hip_fp8_e4m3_fnuz*)a.data_ptr(), (__hip_fp8_e4m3_fnuz*)b.data_ptr(), 
    as.data_ptr<float>(), bs.data_ptr<float>(), (__hip_bfloat16*)c.data_ptr(), m, n, k);
    //C10_CUDA_CHECK(cudaGetLastError());
}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='fp8_mm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['fp8_mm'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"],
)


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    module.fp8_mm(a, b, a_scale, b_scale, c)
    return c

