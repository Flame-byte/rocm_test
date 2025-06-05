#include "softmax.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#ifndef BLOCK_DIM_Y
#define BLOCK_DIM_Y 1024
#endif

#ifndef URF
#define URF 8
#endif

// Define CUDA-style __shfl_xor_sync in terms of HIP __shfl_xor
#ifndef __shfl_xor_sync
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask)
#endif

// template <typename scalar_t>
// __global__ void softmax_kernel(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
// {
//   int row = blockIdx.x;
//   int ty = threadIdx.y;
//   int warp_id = ty/32;
//   __shared__ float reduction_max[BLOCK_DIM_Y/32]; 
//   __shared__ float reduction_div[BLOCK_DIM_Y/32]; 
//   if (row < h)
//   {
//     float maxval = 0;
//     float divisor = 0;
//     float old_maxval = 0;
// #pragma unroll URF
//     for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
//     {
//         float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
//         maxval = fmaxf(maxval, val.x);
//         maxval = fmaxf(maxval, val.y);
//         maxval = fmaxf(maxval, val.z);
//         maxval = fmaxf(maxval, val.w);
//         if (maxval > old_maxval)
//         {
//           divisor *= __expf(old_maxval - maxval);
//           old_maxval = maxval;
//         }
//         divisor += __expf(val.x - maxval);
//         divisor += __expf(val.y - maxval);
//         divisor += __expf(val.z - maxval);
//         divisor += __expf(val.w - maxval);
//     }
//     float incoming_divisor = 0;
//     float incoming_maxval = 0;
// #pragma unroll URF
//     for (int mask = 16; mask>0; mask/=2)
//     {
//       incoming_maxval = __shfl_xor_sync(0xffffffff, maxval, mask, 32);
//       incoming_divisor = __shfl_xor_sync(0xffffffff, divisor, mask, 32);
//       if (incoming_maxval > maxval)
//       {
//         divisor *= __expf(maxval - incoming_maxval);
//         maxval = incoming_maxval;
//       }
//       else 
//       {
//         incoming_divisor *= __expf(incoming_maxval - maxval);
//       }
//       divisor += incoming_divisor;
//     }

//     if (ty%32 == 0)
//     {
//       reduction_max[warp_id] = maxval;
//       reduction_div[warp_id] = divisor;
//     }
//     __syncthreads();
//     if (warp_id == 0)
//     {
//         maxval = ty < BLOCK_DIM_Y/32 ? reduction_max[ty] : 0;
//         divisor = ty < BLOCK_DIM_Y/32 ? reduction_div[ty] : 0;
// #pragma unroll URF
//         for (int mask = 16; mask>0; mask/=2)
//         {
//           incoming_maxval = __shfl_xor_sync(0xffffffff, maxval, mask, 32);
//           incoming_divisor = __shfl_xor_sync(0xffffffff, divisor, mask, 32);
//           if (incoming_maxval > maxval)
//           {
//             divisor *= __expf(maxval - incoming_maxval);
//             maxval = incoming_maxval;
//           }
//           else 
//           {
//             incoming_divisor *= __expf(incoming_maxval - maxval);
//           }
//           divisor += incoming_divisor;
//         }
//     }
//     if (ty == 0)
//     {
//         reduction_max[0] = maxval;
//         reduction_div[0] = divisor;
//     }
//     __syncthreads();
//     maxval = reduction_max[0];
//     divisor = reduction_div[0];

// #pragma unroll URF
//     for (int i = ty; i<w/4; i+=BLOCK_DIM_Y)
//     {
//         float4 val = reinterpret_cast<float4*>(&a[row*w + i*4])[0];
//         val.x = __expf(val.x-maxval)/divisor;
//         val.y = __expf(val.y-maxval)/divisor;
//         val.z = __expf(val.z-maxval)/divisor;
//         val.w = __expf(val.w-maxval)/divisor;
//         reinterpret_cast<float4*>(&b[row*w + i*4])[0] = val;
//     }
//   }
// }

template <typename scalar_t>
__global__ void softmax_kernel(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
    int row = blockIdx.x;
    int ty = threadIdx.y;
    int warp_id = ty / 32;
    int lane_id = ty % 32;

    __shared__ float reduction_max[BLOCK_DIM_Y / 32];
    __shared__ float reduction_div[BLOCK_DIM_Y / 32];

    if (row < h) {
        // 1. 找 maxval
        float maxval = -INFINITY;
        for (int i = ty; i < w; i += BLOCK_DIM_Y) {
            float v = __half2float(a[row * w + i]);
            maxval = fmaxf(maxval, v);
        }
        // warp归约
        for (int mask = 16; mask > 0; mask /= 2)
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
        if (lane_id == 0)
            reduction_max[warp_id] = maxval;
        __syncthreads();
        if (warp_id == 0) {
            maxval = (ty < BLOCK_DIM_Y / 32) ? reduction_max[ty] : -INFINITY;
            for (int mask = 16; mask > 0; mask /= 2)
                maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, mask, 32));
            if (ty == 0)
                reduction_max[0] = maxval;
        }
        __syncthreads();
        maxval = reduction_max[0];

        // 2. 计算 exp(x - maxval) 并归约求和
        float divisor = 0.f;
        for (int i = ty; i < w; i += BLOCK_DIM_Y) {
            float v = __half2float(a[row * w + i]);
            divisor += __expf(v - maxval);
        }
        for (int mask = 16; mask > 0; mask /= 2)
            divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
        if (lane_id == 0)
            reduction_div[warp_id] = divisor;
        __syncthreads();
        if (warp_id == 0) {
            divisor = (ty < BLOCK_DIM_Y / 32) ? reduction_div[ty] : 0.f;
            for (int mask = 16; mask > 0; mask /= 2)
                divisor += __shfl_xor_sync(0xffffffff, divisor, mask, 32);
            if (ty == 0)
                reduction_div[0] = divisor;
        }
        __syncthreads();
        divisor = reduction_div[0];

        // 3. 写回
        for (int i = ty; i < w; i += BLOCK_DIM_Y) {
            float v = __half2float(a[row * w + i]);
            b[row * w + i] = __float2half(__expf(v - maxval) / divisor);
        }
    }
}

// Wrapper for the softmax kernel
extern "C" void softmax(__half* a, __half* b, int w, int h, hipStream_t stream) {

    dim3 grid(h);
    dim3 block(1, BLOCK_DIM_Y);
    size_t shared_mem = (BLOCK_DIM_Y/32) * sizeof(float);
    hipLaunchKernelGGL((softmax_kernel<__half>), grid, block, shared_mem, stream, a, b, w, h);
}

// static float half_to_float(__half h) { return __half2float(h); }
// static __half float_to_half(float f) { return __float2half(f); }

// int main() {
//     const int w = 16;
//     const int h = 4;
//     const int size = w * h;

//     // Host input and output
//     std::vector<__half> h_a(size);
//     std::vector<__half> h_b(size);
//     for (int i = 0; i < size; ++i) {
//         h_a[i] = float_to_half(sin(i * 0.1f));
//     }

//     // Create HIP stream
//     hipStream_t stream;
//     hipStreamCreate(&stream);

//     // Allocate device memory
//     __half* d_a;
//     __half* d_b;
//     hipMalloc(&d_a, size * sizeof(__half));
//     hipMalloc(&d_b, size * sizeof(__half));

//     // Copy input to device
//     hipMemcpyAsync(d_a, h_a.data(), size * sizeof(__half), hipMemcpyHostToDevice, stream);
//     hipStreamSynchronize(stream);

//     // Launch softmax
//     softmax(d_a, d_b, w, h, stream);

//     // Copy output back to host
//     hipMemcpyAsync(h_b.data(), d_b, size * sizeof(__half), hipMemcpyDeviceToHost, stream);
//     hipStreamSynchronize(stream);

//     // CPU reference computation
//     std::vector<float> ref(size);
//     for (int row = 0; row < h; ++row) {
//         int offset = row * w;
//         float maxv = -INFINITY;
//         for (int col = 0; col < w; ++col) {
//             float v = half_to_float(h_a[offset + col]);
//             maxv = std::max(maxv, v);
//         }
//         float sum = 0.0f;
//         for (int col = 0; col < w; ++col) {
//             float v = expf(half_to_float(h_a[offset + col]) - maxv);
//             ref[offset + col] = v;
//             sum += v;
//         }
//         for (int col = 0; col < w; ++col) {
//             ref[offset + col] /= sum;
//         }
//     }

//     // Compare results
//     bool ok = true;
//     for (int i = 0; i < size; ++i) {
//         float gpu = half_to_float(h_b[i]);
//         float cpu = ref[i];
//         if (fabs(gpu - cpu) > 1e-3f) {
//             std::cout << "Mismatch at index " << i << ": GPU=" << gpu << ", CPU=" << cpu << std::endl;
//             ok = false;
//             break;
//         }
//     }

//     if (ok) {
//         std::cout << "Softmax results match CPU reference!" << std::endl;
//     } else {
//         std::cout << "Softmax results do NOT match CPU reference!" << std::endl;
//     }

//     // Cleanup
//     hipFree(d_a);
//     hipFree(d_b);
//     hipStreamDestroy(stream);

// }