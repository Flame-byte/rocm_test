#!/bin/bash

# Set up build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Run the test binary
echo "Running SiLU activation test..."
./test_silu

# Return to the original directory
cd .. 