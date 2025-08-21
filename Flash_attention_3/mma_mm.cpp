// Wave Matrix Multiply Accumulate (WMMA) using HIP compiler intrinsic
// Does a matrix multiplication of two 16x16, fp16 matrices, and stores them into a 16x16 fp16 result matrix

#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
//#include <rocwmma/rocwmma.hpp>

using namespace std;

// using rocwmma::bfloat16_t;

// auto fragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bfloat16_t, rocwmma::row_major>();
// auto fragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bfloat16_t, rocwmma::row_major>();
// auto fragC = rocwmma::fragment<rocwmma::matrix_c, 16, 16, 16, bfloat16_t, rocwmma::row_major>();



// Use half16 as an alias of the internal clang vector type of 16 fp16 values
// typedef _Float16 half16 __attribute__((ext_vector_type(16)));

// typedef hip_bfloat16 bf16x16 __attribute__((ext_vector_type(16)));

typedef uint16_t bf16x16 __attribute__((ext_vector_type(16)));

__global__ void wmma_matmul(__hip_bfloat16* a, __hip_bfloat16* b, __hip_bfloat16* c)
{
    const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lIdx = threadIdx.x;

    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and b
    // a_frag will store one column of the 16x16 matrix A tile
    // b_frag will store one row of the 16x16 matrix B tile
    bf16x16 a_frag;
    bf16x16 b_frag;
    // initialize c fragment to 0
    bf16x16 c_frag = {};

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    const int lane = lIdx % 16;
    // const int x = lIdx % 8;
    // const int y = lIdx / 8;

    for (int ele = 0; ele < 16; ++ele)
    {
        // b_frag[ele] = b[16*ele + lane];
       __hip_bfloat16_raw br = b[16*lane + ele];
       b_frag[ele] = (uint16_t)br.x;
    }

    for (int ele = 0; ele < 16; ++ele)
    {
        __hip_bfloat16_raw ar = a[16*lane + ele];
        a_frag[ele] = (uint16_t)ar.x;
    }


    // call the WMMA intrinsic with OPSEL set to "false"
    c_frag = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(a_frag, b_frag, c_frag, true);

    for (int ele = 0; ele < 8; ++ele)
    {
        const int r = ele * 2 + (lIdx / 16);
        // store results from unpacked c_frag output
        uint16_t raw = (uint16_t)c_frag[ele * 2 + 1];
        __hip_bfloat16_raw cr{raw};
        c[16 * r + lane] = __hip_bfloat16(cr);
        // c[16 * r + lane] = c_frag[ele*2];
        // if OPSEL was set to "true", the line above would instead be
        // c[16 * r + lane] = c_frag[ele*2 + 1];
    }

}

void cpu_matmul(__hip_bfloat16* a, __hip_bfloat16* b, __hip_bfloat16* c, int m, int n, int k) {
    // Perform matrix multiplication: C = A * B
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k_idx = 0; k_idx < k; ++k_idx) {
                // c[i][j] += a[i][k] * b[k][j]
                acc += __bfloat162float(a[i * k + k_idx]) * __bfloat162float(b[j * k + k_idx]);
            }
            c[i * n + j] = __float2bfloat16(acc);
        }
    }
}

int main(int argc, char* argv[])

{
    __hip_bfloat16 a[16 * 16] = {};
    __hip_bfloat16 b[16 * 16] = {};
    __hip_bfloat16 c[16 * 16] = {};
    __hip_bfloat16 c_h[16 * 16] = {};
    __hip_bfloat16 *a_gpu, *b_gpu, *c_gpu;
    hipMalloc(&a_gpu, 16*16 * sizeof(__hip_bfloat16));
    hipMalloc(&b_gpu, 16*16 * sizeof(__hip_bfloat16));
    hipMalloc(&c_gpu, 16*16 * sizeof(__hip_bfloat16));

    // fill in some data into matrices A and B
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            a[i * 16 + j] = (__hip_bfloat16)(j/(i+1));
            b[i * 16 + j] = (__hip_bfloat16)(j/(i+1));
        }
    }

    hipMemcpy(a_gpu, a, (16*16) * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
    hipMemcpy(b_gpu, b, (16*16) * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
    hipMemcpy(c_gpu, c, (16*16) * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);

    cpu_matmul(a, b, c_h, 16, 16, 16);

    wmma_matmul<<<dim3(1), dim3(32, 1, 1), 0, 0>>>(a_gpu, b_gpu, c_gpu);

    hipMemcpy(c, c_gpu, (16 * 16) * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    hipFree(a_gpu);
    hipFree(b_gpu);
    hipFree(c_gpu);

    printf("CPU result:\n");
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            printf("%f ", (float)c_h[i * 16 + j]);
        }
        printf("\n");
    }

    printf("GPU result:\n");
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            printf("%f ", (float)c[i * 16 + j]);
        }
        printf("\n");
    }   

    return 0;
}


// // Wave Matrix Multiply Accumulate (WMMA) using HIP compiler intrinsic
// // Does a matrix multiplication of two 16x16, fp16 matrices, and stores them into a 16x16 fp16 result matrix

// #include <iostream>
// #include <hip/hip_runtime.h>
// #include <hip/hip_fp16.h>
// #include <hip/hip_bf16.h>


// using namespace std;

// // Use half16 as an alias of the internal clang vector type of 16 fp16 values
// typedef _Float16 half16 __attribute__((ext_vector_type(16)));

// // __global__ void wmma_matmul(__half* a, __half* b, __half* c)
// // {
// //     const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
// //     const int lIdx = threadIdx.x;

// //     // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and b
// //     // a_frag will store one column of the 16x16 matrix A tile
// //     // b_frag will store one row of the 16x16 matrix B tile
// //     half16 a_frag;
// //     half16 b_frag;
// //     // initialize c fragment to 0
// //     half16 c_frag = {};

// //     // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
// //     const int lane = lIdx % 16;

// //     for (int ele = 0; ele < 16; ++ele)
// //     {
// //         //b_frag[ele] = b[16*ele + lane];
// //         b_frag[ele] = b[16*lane + ele];
// //     }

// //     for (int ele = 0; ele < 16; ++ele)
// //     {
// //         a_frag[ele] = a[16 * lane + ele];
// //     }

// //     // call the WMMA intrinsic with OPSEL set to "false"
// //     c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

// //     for (int ele = 0; ele < 8; ++ele)
// //     {
// //         const int r = ele * 2 + (lIdx / 16);
// //         // store results from unpacked c_frag output
// //         c[16 * r + lane] = c_frag[ele*2];
// //         // if OPSEL was set to "true", the line above would instead be
// //         // c[16 * r + lane] = c_frag[ele*2 + 1];
// //     }

// // }

// // void cpu_matmul(__half* a, __half* b, __half* c, int m, int n, int k) {
// //     // Perform matrix multiplication: C = A * B
// //     for (int i = 0; i < m; ++i) {
// //         for (int j = 0; j < n; ++j) {
// //             for (int k_idx = 0; k_idx < k; ++k_idx) {
// //                 // c[i][j] += a[i][k] * b[k][j]
// //                 c[i * n + j] += a[i * k + k_idx] * b[k_idx * n + j];
// //             }
// //         }
// //     }
// // }

// void cpu_matmul(__half* a, __half* b, __half* c, int m, int n, int k) {
//     // Perform matrix multiplication: C = A * B
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             for (int k_idx = 0; k_idx < k; ++k_idx) {
//                 // c[i][j] += a[i][k] * b[k][j]
//                 c[i * n + j] += a[i * k + k_idx] * b[j * k + k_idx];
//             }
//         }
//     }
// }

// int main(int argc, char* argv[])

// {
//     __half a[16 * 16] = {};
//     __half b[16 * 16] = {};
//     __half c_d[16 * 16] = {};
//     __half c_h[16 * 16] = {};
//     __half *a_gpu, *b_gpu, *c_gpu;
//     hipMalloc(&a_gpu, 16*16 * sizeof(__half));
//     hipMalloc(&b_gpu, 16*16 * sizeof(__half));
//     hipMalloc(&c_gpu, 16*16 * sizeof(__half));
//     // fill in some data into matrices A and B
//     for (int i = 0; i < 16; ++i)
//     {
//         for (int j = 0; j < 16; ++j)
//         {
//             a[i * 16 + j] = (__half)(j/(i+1));
//             b[i * 16 + j] = (__half)(j/(i+1));
//         }
//     }

//     printf("A:\n");
//     for (int i = 0; i < 16; ++i)
//     {
//         for (int j = 0; j < 16; ++j)
//         {
//             printf("%f ", (float)a[i * 16 + j]);
//         }
//         printf("\n");
//     }
//     printf("B:\n");
//     for (int i = 0; i < 16; ++i)
//     {
//         for (int j = 0; j < 16; ++j)
//         {   
//             printf("%f ", (float)b[i * 16 + j]);
//         }
//         printf("\n");
//     }

//     hipMemcpy(a_gpu, a, (16*16) * sizeof(__half), hipMemcpyHostToDevice);
//     hipMemcpy(b_gpu, b, (16*16) * sizeof(__half), hipMemcpyHostToDevice);
//     hipMemcpy(c_gpu, c_d, (16*16) * sizeof(__half), hipMemcpyHostToDevice);

//     cpu_matmul(a, b, c_h, 16, 16, 16);

//     wmma_matmul<<<dim3(1), dim3(32, 1, 1), 0, 0>>>(a_gpu, b_gpu, c_gpu);

//     hipMemcpy(c_d, c_gpu, (16 * 16) * sizeof(__half), hipMemcpyDeviceToHost);

//     hipFree(a_gpu);
//     hipFree(b_gpu);
//     hipFree(c_gpu);

//     printf("GPU result:\n");
//     for (int i = 0; i < 16; ++i)
//     {
//         for (int j = 0; j < 16; ++j)
//         {
//             printf("%f ", (float)c_d[i * 16 + j]);
//         }
//         printf("\n");
//     }

//     printf("CPU result:\n");
//     for (int i = 0; i < 16; ++i)
//     {
//         for (int j = 0; j < 16; ++j)
//         {
//             printf("%f ", (float)c_h[i * 16 + j]);
//         }
//         printf("\n");
//     }

//     return 0;
// }