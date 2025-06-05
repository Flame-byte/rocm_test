#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__global__ void add(int n, int m, int k, float* A, float* B, float* C)
{
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int y_id = blockIdx.y * blockDim.y + threadIdx.y;
    int z_id = blockIdx.z * blockDim.z + threadIdx.z;

    if (x_id < n && y_id < m && z_id < k)
    {
        // 计算三维数组A中的索引
        int idx_A = x_id * m * k + y_id * k + z_id;
        
        // 将二维数组B映射到三维 - B的元素在每个x维度上重复使用
        int idx_B = y_id * k + z_id;
        
        // 将一维数组C映射到三维 - C的元素在每个x和y维度上重复使用
        int idx_C = z_id;
        
        // 将B和C对应位置的值相加，然后存储到A中
        A[idx_A] = A[idx_A] + B[idx_B] + C[idx_C];
    }

    __shfl_dowm();

    printf("x_id=%d, y_id=%d, z_id=%d\n", x_id, y_id, z_id);
}

int main()
{
    //定义一个三维数组
    int n = 3;
    int m = 4;
    int k = 5;

    std::vector<float> h_A(n * m * k);

    for (int i = 0; i < n * m * k; i++)
    {
        h_A[i] = i + 1.0f;
    }

    //定义一个二维数组
    std::vector<float> h_B(m * k);

    for (int i = 0; i < m * k; i++)
    {
        h_B[i] = i + 1.0f;
    }
    
    //定义一个一维数组
    std::vector<float> h_C(k);

    for (int i = 0; i < k; i++)
    {
        h_C[i] = i + 1.0f;
    }

    // Declare device pointers
    float *d_A, *d_B, *d_C;

    hipMalloc(&d_A, sizeof(float) * n * m * k);
    hipMalloc(&d_B, sizeof(float) * m * k);
    hipMalloc(&d_C, sizeof(float) * k);

    hipMemcpy(d_A, h_A.data(), sizeof(float) * n * m * k, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sizeof(float) * m * k, hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), sizeof(float) * k, hipMemcpyHostToDevice);

    dim3 block(3,4,5);
    dim3 grid(1,1,1);

    hipLaunchKernelGGL(add, grid, block, 0, 0, n, m, k, d_A, d_B, d_C);

    hipMemcpy(h_A.data(), d_A, sizeof(float) * n * m * k, hipMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int l = 0; l < k; l++)
            {
                printf("h_A[%d]=%f\n", i * m * k + j * k + l, h_A[i * m * k + j * k + l]);
            }
        }
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    return 0;
}