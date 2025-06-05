#include <hip/hip_runtime.h> // HIP运行时库，用于GPU计算
#include <iostream>        // 标准输入输出流
#include <vector>          // 标准向量容器
#include <rocblas/rocblas.h>
#include <hip/hip_fp16.h>

#define HIPCHECK(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        printf("HIP error: %s\n", hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define ROCBLAS_CHECK(x) do { \
    rocblas_status err = x; \
    if (err != rocblas_status_success) { \
        printf("ROCBLAS error: %s\n", rocblas_status_to_string(err)); \
        exit(1); \
    } \
} while(0)

int main() {

    // 定义序列长度和嵌入维度
    int seq_len = 2048; // 序列长度，表示输入序列中的token数量
    int dim = 4096;     // 嵌入维度，表示每个token的特征维度

    // 定义矩阵维度
    int m = 4096;  // 权重矩阵的行数
    int n = 4096;  // 权重矩阵的列数
    
    // 定义矩阵乘法的缩放因子
    __half alpha = __float2half(1.0f); // 矩阵乘法的乘数因子
    __half beta = __float2half(0.0f);  // 累加结果的因子

    __half* h_embedded_matrix;
    __half* h_wq;
    __half* h_result;

    h_embedded_matrix = (__half*)malloc(seq_len * dim * sizeof(__half));
    h_wq = (__half*)malloc(n * m * sizeof(__half));
    h_result = (__half*)malloc(seq_len * n * sizeof(__half));

    for(int i = 0; i < seq_len * dim; i++)
    {
        h_embedded_matrix[i] = __float2half(1.0f);
    }

    for(int i = 0; i < n * m; i++)
    {
        h_wq[i] = __float2half(2.0f);
    }

    __half* d_embedded_matrix;
    __half* d_wq;
    __half* d_result;

    HIPCHECK(hipMalloc(&d_embedded_matrix, seq_len * dim * sizeof(__half)));
    HIPCHECK(hipMalloc(&d_wq, n * m * sizeof(__half)));
    HIPCHECK(hipMalloc(&d_result, seq_len * n * sizeof(__half)));

    // hipMemcpy(d_embedded_matrix, h_embedded_matrix, seq_len * dim * sizeof(__half), hipMemcpyHostToDevice);
    // hipMemcpy(d_wq, h_wq, n * m * sizeof(__half), hipMemcpyHostToDevice);

    //Create rocblas handle
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
    
    // Copy the matrix from host memory to device memory
    ROCBLAS_CHECK(rocblas_set_matrix(seq_len, dim, sizeof(__half), h_embedded_matrix, seq_len, d_embedded_matrix, seq_len));
    ROCBLAS_CHECK(rocblas_set_matrix(m, n, sizeof(__half), h_wq, m, d_wq, m));

    // C = α * op(A) * op(B) + β * C
    ROCBLAS_CHECK(rocblas_gemm_ex(handle, 
                    rocblas_operation_none,       // The A matrix is ​​not transposed
                    rocblas_operation_transpose,  // B matrix is transposed
                    seq_len,                      // The number of rows of A matrix = the number of rows of C matrix
                    n,                           // The number of columns of B matrix = the number of columns of C matrix
                    m,                           // The number of columns of A matrix = the number of rows of B matrix
                    &alpha,                       // Scalar multiplication factor α
                    d_embedded_matrix,            // The A matrix (embedded matrix)
                    rocblas_datatype_f16_r,       // The data type of the A matrix (half precision)
                    seq_len,                      // The main dimension of the A matrix (modified to seq_len)
                    d_wq,                         // The B matrix (WQ weight matrix)
                    rocblas_datatype_f16_r,       // The data type of the B matrix (half precision)
                    m,                            // The main dimension of the B matrix (modified to m)
                    &beta,                        // Scalar addition factor β
                    d_result,                     // The output matrix C
                    rocblas_datatype_f16_r,       // The data type of the C matrix (half precision)
                    seq_len,                      // The main dimension of the C matrix
                    d_result,                     // The matrix C space for calculation
                    rocblas_datatype_f16_r,       // The data type of the calculation matrix C
                    seq_len,                      // The main dimension of the calculation matrix C
                    rocblas_datatype_f16_r,       // The data type of the internal calculation (modified to explicitly use float32)
                    rocblas_gemm_algo_standard,   // The matrix multiplication algorithm
                    0,                            // The details of the algorithm
                    0));                           // The reserved parameters

    // Copy the result from the device memory to the host memory
    ROCBLAS_CHECK(rocblas_get_matrix(seq_len, n, sizeof(__half), d_result, seq_len, h_result, seq_len));

    // hipMemcpy(h_result, d_result, seq_len * n * sizeof(__half), hipMemcpyDeviceToHost);
    
    // 打印结果矩阵的部分值，每4096个元素取10个值打印
    for (int i = 0; i < seq_len && i < 10; i++) {
        printf("Row %d: ", i);
        for(int j = 0; j < 10; j++)
        {
            printf("result[%d,%d]=%f ", i, j, __half2float(h_result[i*n + j])); // 将half类型转换为float后打印
        }
        printf("\n");
    }

    //清理资源
    free(h_embedded_matrix);       // 释放主机内存
    free(h_wq);                    // 释放主机内存
    free(h_result);                // 释放主机内存
    
    rocblas_destroy_handle(handle); // 销毁rocBLAS句柄
    hipFree(d_result);              // 释放结果矩阵的设备内存
    hipFree(d_embedded_matrix);     // 释放嵌入矩阵的设备内存
    hipFree(d_wq);                  // 释放权重矩阵的设备内存

    return 0;
}