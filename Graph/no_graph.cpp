#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>

#define HIP_CHECK(expression)                \
{                                            \
    const hipError_t status = expression;    \
    if(status != hipSuccess){                \
            std::cerr << "HIP error "        \
                << status << ": "            \
                << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;    \
    }                                        \
}


__global__ void kernelA(double* arrayA, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){arrayA[x] *= 2.0;}
};
__global__ void kernelB(int* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){arrayB[x] = 3;}
};
__global__ void kernelC(double* arrayA, const int* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){arrayA[x] += arrayB[x];}
};

struct set_vector_args{
    std::vector<double>& h_array;
    double value;
};

void set_vector(void* args){
    set_vector_args h_args{*(reinterpret_cast<set_vector_args*>(args))};

    std::vector<double>& vec{h_args.h_array};
    vec.assign(vec.size(), h_args.value);
}

int main(){
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 65536;//1U << 20;

    double* d_arrayA;
    int* d_arrayB;
    std::vector<double> h_array(arraySize);
    constexpr double initValue = 2.0;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // 普通方式：直接顺序执行
    int repeat = 100; // 重复次数
    for (int i = 0; i < repeat; ++i) {
        HIP_CHECK(hipMallocAsync(&d_arrayA, arraySize*sizeof(double), stream));
        HIP_CHECK(hipMallocAsync(&d_arrayB, arraySize*sizeof(int), stream));

        set_vector_args args{h_array, initValue};
        HIP_CHECK(hipLaunchHostFunc(stream, set_vector, &args));

        HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array.data(), arraySize*sizeof(double), hipMemcpyHostToDevice, stream));

    
        kernelA<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, arraySize);
        kernelB<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayB, arraySize);
        kernelC<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySize);
        HIP_CHECK(hipStreamSynchronize(stream)); // 每轮同步，确保顺序
    

        HIP_CHECK(hipMemcpyAsync(h_array.data(), d_arrayA, arraySize*sizeof(*d_arrayA), hipMemcpyDeviceToHost, stream));
        

        HIP_CHECK(hipFreeAsync(d_arrayA, stream));
        HIP_CHECK(hipFreeAsync(d_arrayB, stream));

    }

    // 验证结果
    constexpr double expected = initValue * 2.0 + 3;
    bool passed = true;
    for(size_t i = 0; i < arraySize; ++i){
            if(h_array[i] != expected){
                    passed = false;
                    std::cerr << "Validation failed! Expected " << expected << " got " << h_array[0] << std::endl;
                    break;
            }
    }
    if(passed){
            std::cerr << "Validation passed." << std::endl;
    }

    HIP_CHECK(hipStreamDestroy(stream));
}