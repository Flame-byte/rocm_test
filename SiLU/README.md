# SiLU (Swish) Activation Function for HIP/ROCm

This repository contains an implementation of the SiLU (Sigmoid Linear Unit) activation function, also known as Swish, using AMD's HIP (Heterogeneous-Computing Interface for Portability) for GPU acceleration.

## Overview

SiLU is defined as:
```
SiLU(x) = x * sigmoid(x)
```

where `sigmoid(x) = 1/(1 + exp(-x))`.

This implementation is optimized for half-precision (`__half`) floating-point operations, making it suitable for deep learning applications that prioritize memory efficiency and speed.

## Requirements

- ROCm (Rock Computing Module) >= 4.0
- CMake >= 3.10
- C++ compiler with C++14 support

## Building the Project

1. Navigate to the project directory:
```bash
cd SiLU
```

2. Create a build directory and enter it:
```bash
mkdir build && cd build
```

3. Configure with CMake:
```bash
cmake ..
```

4. Build the project:
```bash
make
```

## Usage

The library provides the following API:

```cpp
// C-compatible wrapper for SiLU activation
extern "C" void silu_activation_wrapper(
    const __half* input,    // Input tensor
    __half* output,         // Output tensor (must be preallocated)
    const int seq_len,      // Sequence length dimension
    const int hidden_dim    // Hidden dimension size
);
```

### Example Usage

A simple example is provided in `test_silu.cpp`. After building, you can run:

```bash
./test_silu
```

This will perform SiLU activation on sample data and print the results.

## Performance Considerations

- The kernel uses a simple 1D thread mapping, which works well for most cases.
- For very large matrices, more sophisticated 2D decomposition might be beneficial.
- The current implementation converts from half to float for the computation and back to half for storage. This ensures numerical stability while maintaining memory efficiency.

## License

This code is provided under the MIT License.

## References

- SiLU/Swish activation: Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).
- ROCm Documentation: https://rocmdocs.amd.com/ 