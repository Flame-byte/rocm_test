#include <hip/hip_runtime.h> // HIP运行时库，用于GPU计算
#include <iostream>        // 标准输入输出流
#include "weight_loader.h" // 权重加载器，用于加载模型权重
#include <vector>          // 标准向量容器
#include <rocblas.h>       // ROCm BLAS库，用于GPU上的线性代数运算

int main() {
    // 初始化WeightLoader，提供模型索引和权重文件的路径
    WeightLoader loader(
        "/home/qin/rocm_test/Embedding/model_index.json", // 模型索引文件路径
        "/home/qin/rocm_test/Embedding/model_weights.bin" // 模型权重二进制文件路径
    );
    

    // 从权重文件加载Query权重矩阵(WQ)
    __half* wq = loader.get_weights("layers.0.attention.wq.weight");

    // 打印WQ权重矩阵的部分值，每4096个元素取10个值打印
    for (int i = 0; i < 40960; i+=4096) {
        for(int j = 0; j < 10; j++)
        {
            printf("wq[%d]=%f ", i+j, __half2float(wq[i+j])); // 将half类型转换为float后打印
        }
        printf("\n");
    }

    // 定义序列长度和嵌入维度
    int seq_len = 2048; // 序列长度，表示输入序列中的token数量
    int dim = 4096;     // 嵌入维度，表示每个token的特征维度

    // 定义矩阵维度
    int m = loader.get_shape("layers.0.attention.wq.weight")[0];  // 权重矩阵的行数
    int n = loader.get_shape("layers.0.attention.wq.weight")[1];  // 权重矩阵的列数
    std::cout << "m: " << m << " n: " << n << std::endl;          // 打印权重矩阵的维度

    
    // 定义矩阵乘法的缩放因子
    __half alpha = __float2half(1.0f); // 矩阵乘法的乘数因子
    __half beta = __float2half(0.0f);  // 累加结果的因子

    __half* embedded_matrix;
    __half* result;

    hipMallocManaged(&embedded_matrix, seq_len * dim * sizeof(__half));
    hipMallocManaged(&result, seq_len * n * sizeof(__half));

    for(int i = 0; i < seq_len * dim; i++)
    {
        embedded_matrix[i] = __float2half(1.0f);
    }

    // // // 声明矩阵相关指针
    // __half* h_embedded_matrix; // 主机端嵌入矩阵
    // __half* d_embedded_matrix; // 设备端嵌入矩阵
    // __half* d_wq;             // 设备端WQ权重矩阵
    // __half* d_result;         // 设备端结果矩阵
    // __half* h_result;         // 主机端结果矩阵
    
    // // 初始化嵌入矩阵（设置所有元素为1.0）
    // h_embedded_matrix = (__half*)malloc(seq_len * dim * sizeof(__half));
    // h_result = (__half*)malloc(seq_len * n * sizeof(__half));

    // for(int i = 0; i < seq_len * dim; i++)
    // {
    //     h_embedded_matrix[i] = __float2half(1.0f); // 设置所有元素为1.0
    // }

    // // 在GPU上分配内存
    // hipMalloc(&d_embedded_matrix, seq_len * dim * sizeof(__half));  // 为嵌入矩阵分配设备内存
    // hipMalloc(&d_wq, n * m * sizeof(__half));                      // 为WQ权重矩阵分配设备内存
    // hipMalloc(&d_result, seq_len * n * sizeof(__half));            // 为结果矩阵分配设备内存 

    // hipMemcpy(d_embedded_matrix, h_embedded_matrix, seq_len * dim * sizeof(__half), hipMemcpyHostToDevice);
    // hipMemcpy(d_wq, h_wq, n * m * sizeof(__half), hipMemcpyHostToDevice);

    // 将矩阵从主机内存复制到设备内存
    // rocblas_set_matrix(seq_len, dim, sizeof(__half), h_embedded_matrix, dim, d_embedded_matrix, dim); // 复制嵌入矩阵
    // rocblas_set_matrix(m, n, sizeof(__half), h_wq, n, d_wq, n);  
    
    hipMallocManaged(&embedded_matrix, seq_len * dim * sizeof(__half));
    hipMallocManaged(&result, seq_len * n * sizeof(__half));

    for(int i = 0; i < seq_len * dim; i++)
    {
        embedded_matrix[i] = __float2half(1.0f);
    }                                     

    // Create rocblas handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    
    // rocblas_set_matrix(seq_len, dim, sizeof(__half), embedded_matrix, seq_len, embedded_matrix, seq_len);
    // rocblas_set_matrix(m, n, sizeof(__half), wq, m, wq, m);
    
    // C = α * op(A) * op(B) + β * C
    rocblas_gemm_ex(handle, 
                    rocblas_operation_none,       // The A matrix is ​​not transposed
                    rocblas_operation_transpose,  // B matrix is transposed
                    seq_len,                      // The number of rows of A matrix = the number of rows of C matrix
                    n,                           // The number of columns of B matrix = the number of columns of C matrix 
                    m,                           // The number of columns of A matrix = the number of rows of B matrix
                    &alpha,                       // Scalar multiplication factor α
                    embedded_matrix,            // The A matrix (embedded matrix)
                    rocblas_datatype_f16_r,       // The data type of the A matrix (half precision)
                    dim,                          // The main dimension of the A matrix
                    wq,                         // The B matrix (WQ weight matrix)
                    rocblas_datatype_f16_r,       // The data type of the B matrix (half precision)
                    n,                            // The main dimension of the B matrix
                    &beta,                        // Scalar addition factor β
                    result,                     // The output matrix C
                    rocblas_datatype_f16_r,       // The data type of the C matrix (half precision)
                    seq_len,                      // The main dimension of the C matrix
                    result,                     // The matrix C space for calculation
                    rocblas_datatype_f16_r,       // The data type of the calculation matrix C
                    seq_len,                      // The main dimension of the calculation matrix C
                    rocblas_datatype_f16_r,       // The data type of the internal calculation (modified to explicitly use float32)
                    rocblas_gemm_algo_standard,   // The matrix multiplication algorithm
                    0,                            // The details of the algorithm
                    0);                           // The reserved parameters

    // Copy the result from the device memory to the host memory
    //rocblas_set_matrix(seq_len, n, sizeof(__half), result, seq_len, h_result, seq_len);
    
    for (int i = 0; i < 40960; i+=4096) {
        for(int j = 0; j < 10; j++)
        {
            printf("result[%d]=%f ", i+j, __half2float(result[i+j])); 
        }
        printf("\n");
    }

    //清理资源
    rocblas_destroy_handle(handle); // 销毁rocBLAS句柄
    loader.free_all();              // 释放加载器占用的所有资源
    hipFree(result);              // 释放结果矩阵的设备内存
    hipFree(embedded_matrix);     // 释放嵌入矩阵的设备内存

    return 0;
}