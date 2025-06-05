#include "softmax.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

static float half_to_float(__half h) { return __half2float(h); }
static __half float_to_half(float f) { return __float2half(f); }

int main() {
    const int w = 16;
    const int h = 4;
    const int size = w * h;

    // Host input and output
    std::vector<__half> h_a(size);
    std::vector<__half> h_b(size);
    for (int i = 0; i < size; ++i) {
        h_a[i] = float_to_half(sin(i * 0.1f));
    }

    // Create HIP stream
    hipStream_t stream;
    hipStreamCreate(&stream);

    // Allocate device memory
    __half* d_a;
    __half* d_b;
    hipMalloc(&d_a, size * sizeof(__half));
    hipMalloc(&d_b, size * sizeof(__half));

    // Copy input to device
    hipMemcpyAsync(d_a, h_a.data(), size * sizeof(__half), hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);

    // Launch softmax
    softmax(d_a, d_b, w, h, stream);

    // Copy output back to host
    hipMemcpyAsync(h_b.data(), d_b, size * sizeof(__half), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);

    // CPU reference computation
    std::vector<float> ref(size);
    for (int row = 0; row < h; ++row) {
        int offset = row * w;
        float maxv = -INFINITY;
        for (int col = 0; col < w; ++col) {
            float v = half_to_float(h_a[offset + col]);
            maxv = std::max(maxv, v);
        }
        float sum = 0.0f;
        for (int col = 0; col < w; ++col) {
            float v = expf(half_to_float(h_a[offset + col]) - maxv);
            ref[offset + col] = v;
            sum += v;
        }
        for (int col = 0; col < w; ++col) {
            ref[offset + col] /= sum;
        }
    }

    // Compare results
    bool ok = true;
    for (int i = 0; i < size; ++i) {
        float gpu = half_to_float(h_b[i]);
        float cpu = ref[i];
        if (fabs(gpu - cpu) > 1e-3f) {
            std::cout << "Mismatch at index " << i << ": GPU=" << gpu << ", CPU=" << cpu << std::endl;
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Softmax results match CPU reference!" << std::endl;
    } else {
        std::cout << "Softmax results do NOT match CPU reference!" << std::endl;
    }

    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipStreamDestroy(stream);

    return ok ? 0 : 1;
} 