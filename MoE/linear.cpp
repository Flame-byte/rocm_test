#include "linear.h"

extern "C" void linear(
    __half* a,
    __half* b,
    __half* d,
    int batch_size,
    int seq_len,
    int d_hidden,
    int d_expert,
    rocblas_handle handle
)
{
    __half alpha = __float2half(1.0f); // 矩阵乘法的乘数因子
    __half beta = __float2half(0.0f);  // 累加结果的因子

    rocblas_gemm_strided_batched_ex(
        handle,
        rocblas_operation_transpose,   // A^T
        rocblas_operation_none,   // B^T
        d_expert,                      // m (n)
        seq_len,                       // n (m)
        d_hidden,                      // k
        &alpha,
        b, rocblas_datatype_f16_r,   // A: B
        d_hidden,                      // lda (B的行数)
        0,//d_hidden * d_expert,           // strideA
        a, rocblas_datatype_f16_r,   // B: A
        d_hidden,                      // ldb (A的列数)
        seq_len * d_hidden,            // strideB
        &beta,
        d, rocblas_datatype_f16_r,   // C
        d_expert,                      // ldc (C的列数)
        seq_len * d_expert,            // strideC
        d, rocblas_datatype_f16_r,   // D
        d_expert,                      // ldd
        seq_len * d_expert,            // strideD
        batch_size,
        rocblas_datatype_f16_r,
        rocblas_gemm_algo_standard,
        0,
        0
    );
}

// extern "C" void linear(
//     __half* a,
//     __half* b,
//     __half* d,
//     int seq_len,
//     int d_hidden,
//     int d_expert,
//     rocblas_handle handle
// )
// {
//     __half alpha = __float2half(1.0f); // 矩阵乘法的乘数因子
//     __half beta = __float2half(0.0f);  // 累加结果的因子

//     // C = α * op(A) * op(B) + β * C
//     rocblas_gemm_ex(handle, 
//                     rocblas_operation_none,       // The A matrix is ​​not transposed
//                     rocblas_operation_none,  // B matrix is transposed
//                     seq_len,                      // The number of rows of A matrix = the number of rows of C matrix
//                     d_expert,                           // The number of columns of B matrix = the number of columns of C matrix 
//                     d_hidden,                           // The number of columns of A matrix = the number of rows of B matrix
//                     &alpha,                       // Scalar multiplication factor α
//                     a,            // The A matrix (embedded matrix)
//                     rocblas_datatype_f16_r,       // The data type of the A matrix (half precision)
//                     dim,                          // The main dimension of the A matrix
//                     wq,                         // The B matrix (WQ weight matrix)
//                     rocblas_datatype_f16_r,       // The data type of the B matrix (half precision)
//                     n,                            // The main dimension of the B matrix
//                     &beta,                        // Scalar addition factor β
//                     result,                     // The output matrix C
//                     rocblas_datatype_f16_r,       // The data type of the C matrix (half precision)
//                     seq_len,                      // The main dimension of the C matrix
//                     result,                     // The matrix C space for calculation
//                     rocblas_datatype_f16_r,       // The data type of the calculation matrix C
//                     seq_len,                      // The main dimension of the calculation matrix C
//                     rocblas_datatype_f16_r,       // The data type of the internal calculation (modified to explicitly use float32)
//                     rocblas_gemm_algo_standard,   // The matrix multiplication algorithm
//                     0,                            // The details of the algorithm
//                     0);                           // The reserved parameters
// }
// #include "linear.h"

// extern "C" void linear(
//     __half* h_a,
//     __half* h_b,
//     __half* h_d,
//     int batch_size,
//     int seq_len,
//     int d_hidden,
//     int d_expert
// )
// {
//     __half* d_a, *d_b, *d_d;
//     hipMalloc(&d_a, seq_len * d_hidden * batch_size * sizeof(__half));
//     hipMalloc(&d_b, d_hidden * d_expert * sizeof(__half));
//     hipMalloc(&d_d, seq_len * d_expert * batch_size * sizeof(__half));

//     __half alpha = __float2half(1.0f); // 矩阵乘法的乘数因子
//     __half beta = __float2half(0.0f);  // 累加结果的因子

//     // 拷贝A: (batch_size, seq_len, d_hidden) 展平成 (batch_size*seq_len, d_hidden)
//     rocblas_set_matrix(seq_len * batch_size, d_hidden, sizeof(__half), h_a, seq_len * batch_size, d_a, seq_len * batch_size);
//     // 拷贝B: (d_hidden, d_expert)
//     rocblas_set_matrix(d_hidden, d_expert, sizeof(__half), h_b, d_hidden, d_b, d_hidden);

//     // // Using hipMemcpy to copy data from host to device
//     // hipMemcpy(d_a, h_a, seq_len * d_hidden * batch_size * sizeof(__half), hipMemcpyHostToDevice);
//     // hipMemcpy(d_b, h_b, d_hidden * d_expert * sizeof(__half), hipMemcpyHostToDevice);

//     rocblas_handle handle;
//     rocblas_create_handle(&handle);

//     //rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

//     // rocblas_gemm_strided_batched_ex(
//     //     handle,
//     //     rocblas_operation_none,        // ROCBLAS_OP_N
//     //     rocblas_operation_none,        // ROCBLAS_OP_N
//     //     seq_len,             // m
//     //     d_expert,            // n
//     //     d_hidden,            // k
//     //     &alpha,              // alpha
//     //     d_a,                 // A
//     //     rocblas_datatype_f16_r, // A_type
//     //     d_hidden,            // lda
//     //     seq_len * d_hidden,  // strideA
//     //     d_b,                 // B
//     //     rocblas_datatype_f16_r, // B_type
//     //     d_expert,            // ldb
//     //     0,                   // strideB
//     //     &beta,               // beta
//     //     d_d,                 // C
//     //     rocblas_datatype_f16_r, // C_type
//     //     d_expert,            // ldc
//     //     seq_len * d_expert,  // strideC
//     //     d_d,                 // D
//     //     rocblas_datatype_f16_r, // D_type
//     //     d_expert,            // ldD
//     //     seq_len * d_expert,  // strideD
//     //     batch_size,          // batch_count
//     //     rocblas_datatype_f16_r,
//     //     rocblas_gemm_algo_standard,
//     //     0,
//     //     rocblas_gemm_flags_none
//     // );

//     rocblas_gemm_strided_batched_ex(
//     handle,
//     rocblas_operation_transpose,   // A^T
//     rocblas_operation_none,   // B^T
//     d_expert,                      // m (n)
//     seq_len,                       // n (m)
//     d_hidden,                      // k
//     &alpha,
//     d_b, rocblas_datatype_f16_r,   // A: B
//     d_hidden,                      // lda (B的行数)
//     0,//d_hidden * d_expert,           // strideA
//     d_a, rocblas_datatype_f16_r,   // B: A
//     d_hidden,                      // ldb (A的列数)
//     seq_len * d_hidden,            // strideB
//     &beta,
//     d_d, rocblas_datatype_f16_r,   // C
//     d_expert,                      // ldc (C的列数)
//     seq_len * d_expert,            // strideC
//     d_d, rocblas_datatype_f16_r,   // D
//     d_expert,                      // ldd
//     seq_len * d_expert,            // strideD
//     batch_size,
//     rocblas_datatype_f16_r,
//     rocblas_gemm_algo_standard,
//     0,
//     0
// );


//     // 拷贝结果回主机
//     rocblas_get_matrix(seq_len * batch_size, d_expert, sizeof(__half), d_d, seq_len * batch_size, h_d, seq_len * batch_size);
//     //hipMemcpy(h_d, d_d, seq_len * d_expert * batch_size * sizeof(__half), hipMemcpyDeviceToHost);

//     hipFree(d_a);
//     hipFree(d_b);
//     hipFree(d_d);
//     rocblas_destroy_handle(handle);
// }




// #include <random>
// #include <cassert>
// #include <cmath>

// float half_to_float(__half h) {
//     return __half2float(h);
// }

// __half float_to_half(float f) {
//     return __float2half(f);
// }

// int main() {
//     // 参数
//     int batch_size = 2;
//     int seq_len = 3;
//     int d_hidden = 4;
//     int d_expert = 5;

//     int a_size = batch_size * seq_len * d_hidden;
//     int b_size = d_hidden * d_expert;
//     int c_size = batch_size * seq_len * d_expert;

//     __half* h_a = (__half*)malloc(a_size * sizeof(__half));
//     __half* h_b = (__half*)malloc(b_size * sizeof(__half));
//     __half* h_c = (__half*)malloc(c_size * sizeof(__half));

//     for (int i = 0; i < a_size; ++i) {
//         h_a[i] = float_to_half(i/10.0f);
//     }
//     for (int i = 0; i < b_size; ++i) {
//         h_b[i] = float_to_half(i/10.0f);
//     }

//     // 调用linear
//     linear(h_a, h_b, h_c, batch_size, seq_len, d_hidden, d_expert);

//     // CPU 计算参考结果
//     // a: (batch_size, seq_len, d_hidden), row-major
//     // b: (d_hidden, d_expert), column-major
//     // c = a * b
//     std::vector<float> ref_c(c_size, 0.0f);
//     for (int bidx = 0; bidx < batch_size; ++bidx) {
//         for (int i = 0; i < seq_len; ++i) {
//             for (int j = 0; j < d_expert; ++j) {
//                 float sum = 0.0f;
//                 for (int k = 0; k < d_hidden; ++k) {
//                     // a: row-major, index = ((bidx * seq_len + i) * d_hidden + k)
//                     float a_val = half_to_float(h_a[(bidx * seq_len + i) * d_hidden + k]);
//                     // b: column-major, index = (k + j * d_hidden)
//                     float b_val = half_to_float(h_b[k + j * d_hidden]);
//                     sum += a_val * b_val;
//                 }
//                 ref_c[(bidx * seq_len + i) * d_expert + j] = sum;
//             }
//         }
//     }

//     // 校验
//     bool ok = true;
//     for (int i = 0; i < c_size; ++i) {
//         float v1 = half_to_float(h_c[i]);
//         float v2 = ref_c[i];
//         if (std::abs(v1 - v2) > 1e-2f) {
//             std::cout << "Mismatch at " << i << ": GPU=" << v1 << " CPU=" << v2 << std::endl;
//             ok = false;
//         }
//     }
//     if (ok) {
//         std::cout << "Linear result matches CPU reference!" << std::endl;
//     } else {
//         std::cout << "Linear result does NOT match CPU reference!" << std::endl;
//     }

//     // 打印 GPU 结果
//     std::cout << "GPU output (all values): ";
//     for (int i = 0; i < c_size; ++i) {
//         std::cout << half_to_float(h_c[i]) << " ";
//     }
//     std::cout << std::endl;

//     // 打印 CPU 参考结果
//     std::cout << "CPU reference output (all values): ";
//     for (int i = 0; i < c_size; ++i) {
//         std::cout << ref_c[i] << " ";
//     }
//     std::cout << std::endl;

//     // 打印部分结果
//     std::cout << "First 5 output values (GPU): ";
//     for (int i = 0; i < std::min(5, c_size); ++i) {
//         std::cout << half_to_float(h_c[i]) << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }