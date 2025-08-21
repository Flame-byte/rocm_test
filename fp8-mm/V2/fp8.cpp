#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_runtime.h>
#include <torch/torch.h>

//定义了一个 SIMD（单指令多数据）向量类型别名，用于高性能并行计算
//__attribute__ 是 GCC 和 Clang 的扩展，用于向编译器提供额外的信息
//__vector_size__ 是 GCC 的扩展，用于指定向量类型的大小
//16 * sizeof(float) 表示向量类型的大小为 16 个 float 类型的大小
//float 表示向量类型中的元素类型
//f32x16 是向量类型别名，表示一个包含 16 个 float 类型的向量
using f32x16 = __attribute__( (__vector_size__(16 * sizeof(float)) )) float;

constexpr const size_t BLOCK = 128;

//__device__ 是 HIP 的扩展，用于指定函数在设备上执行
//static 表示函数是静态的，即在编译时就已经确定，不需要在运行时分配内存
//uint64_t 是 64 位无符号整数类型
//pack_u16x8_to_u64 函数将 8 个 16 位无符号整数打包成一个 64 位无符号整数
//const uint16_t* src 是输入参数，表示一个包含 8 个 16 位无符号整数的数组
//reinterpret_cast<uint64_t*>(dst) 将 dst 数组转换为 uint64_t 类型的指针
//return *reinterpret_cast<uint64_t*>(dst) 返回转换后的 uint64_t 类型的值
__device__ static uint64_t pack_u16x8_to_u64(const uint16_t* src)
{
    uint16_t dst[] = { src[1], src[3], src[5], src[7] };
    return *reinterpret_cast<uint64_t*>(dst);
}

__global__ void custom_kernel(const uint64_t* cache, const float* as, const float* bs, 
                   __hip_bfloat16* c, size_t m, size_t n, size_t k, size_t col, size_t offset) {
    const uint64_t* a = cache;
    const uint64_t* b = cache + offset;
    size_t bx = blockIdx.x * 2 + (threadIdx.y & 1); //把 block 在 m 方向再切成两半
    size_t by = blockIdx.y * 2 + (threadIdx.y >> 1);    //把 block 在 n 方向再切成两半
    size_t cx = 32 * bx; //当前线程要处理的元素的地址
    size_t cy = 32 * by; //当前线程要处理的元素的地址
    if(cx >= m || cy >= n) return; //如果当前线程要处理的元素的地址超出 m 或 n 的范围，则返回
    size_t tx = threadIdx.x & 31;
    size_t ty = threadIdx.x >> 5 & 1;
    size_t sn = (n + BLOCK - 1) >> 7;
    bs += cy >> 7;
    as += cx + tx;
    a += bx * col + threadIdx.x;    //这边的col=k*4,相当于一行k有多少bit
    b += by * col + threadIdx.x;
    f32x16 result = { };
    for(size_t i = 0; i < k; i += BLOCK) {
        const auto* ap = a + i * 4;
        const auto* bp = b + i * 4;
        f32x16 block_result = { };
        block_result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(bp[0], ap[0], block_result, 0, 0, 0);
        block_result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(bp[0x40], ap[0x40], block_result, 0, 0, 0);
        block_result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(bp[0x80], ap[0x80], block_result, 0, 0, 0);
        block_result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(bp[0xc0], ap[0xc0], block_result, 0, 0, 0);
        block_result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(bp[0x100], ap[0x100], block_result, 0, 0, 0);
        block_result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(bp[0x140], ap[0x140], block_result, 0, 0, 0);
        block_result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(bp[0x180], ap[0x180], block_result, 0, 0, 0);
        block_result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(bp[0x1c0], ap[0x1c0], block_result, 0, 0, 0);
        result += block_result * bs[(i >> 7) * sn] * as[(i >> 7) * m];
    }
    const auto* src = reinterpret_cast<const uint16_t(*)[8]>(&result);
    uint64_t* dst = reinterpret_cast<uint64_t*>(c + (tx + cx) * n + cy + ty * 4);
    dst[0] = pack_u16x8_to_u64(src[0]);
    dst[2] = pack_u16x8_to_u64(src[1]);
    dst[4] = pack_u16x8_to_u64(src[2]);
    dst[6] = pack_u16x8_to_u64(src[3]);
}

__global__ void permute_kernel(const __hip_fp8_e4m3_fnuz* src, const __hip_fp8_e4m3_fnuz* src1, uint64_t* dst,
    size_t m, size_t n, size_t k, size_t col, size_t as0, size_t as1, size_t bs0, size_t bs1, size_t next) {
    next = blockIdx.x >= next ? next : 0;   //前面若干 block 打包 A，剩下的打包 B
    m = next ? n : m;  //当next=0时，处理的是
    as0 = next ? bs0 : as0; //as0=m,bs0=n
    as1 = next ? bs1 : as1; //as1=k,bs1=k
    src = next ? src1 : src; //src=a,src1=b 
    size_t cx = (blockIdx.x - next) * 32 + (threadIdx.x & 31);
    size_t cy = (blockIdx.y * 2 + (threadIdx.x >> 5 & 1)) * 8;
    uint64_t value = 0;
    if (cx < m && cy < k) {
        src += cy * as1 + cx * as0; //当前线程要处理的元素的地址
        __hip_fp8_e4m3_fnuz buffer[8] = { src[0], src[as1], src[2 * as1], src[3 * as1],
            src[4 * as1], src[5 * as1], src[6 * as1], src[7 * as1]};
        value = *reinterpret_cast<uint64_t*>(buffer);
    }
    dst[blockIdx.x * col + blockIdx.y * 64 + threadIdx.x] = value;
}

void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    size_t m = a.size(0);
    size_t n = b.size(0);
    size_t k = a.size(1);
    size_t next = (m + 127) / 128 * 4;  //m方向上要分多少个大小为32的块，每个块包含32个元素，大小为32Bytes
    size_t row = (n + 127) / 128 * 4 + next;    //n方向上要分多少个大小为32的块，每个块包含32个元素，大小为32Bytes
    size_t col = (k + 127) / 128 * 512;  //
    auto options = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA);
    torch::Tensor cache = torch::empty(row * col, options); //在 GPU 上分配一个一维张量，长度为 row * col，数据类型为 uint64_t
    permute_kernel<<<dim3(row, col / 64), dim3(64), 0, 0>>>((__hip_fp8_e4m3_fnuz*)a.data_ptr(),  (__hip_fp8_e4m3_fnuz*)b.data_ptr(),
        cache.data_ptr<uint64_t>(), m, n, k, col, a.stride(0), a.stride(1), b.stride(0), b.stride(1), next);
    custom_kernel<<<dim3((m+63)/64, (n+63)/64), dim3(64, 4), 0, 0>>> (cache.data_ptr<uint64_t>(),
        as.data_ptr<float>(), bs.data_ptr<float>(), (__hip_bfloat16*)c.data_ptr(), m, n, k, col, next * col);
}