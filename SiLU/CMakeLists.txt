cmake_minimum_required(VERSION 3.10)
project(SiLU_HIP_Activation)

# Find the HIP package
find_package(hip REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add HIP flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HIPCC_FLAGS}")

# Add the SiLU library
add_library(silu_activation SHARED
    SiLU.cpp
)
target_include_directories(silu_activation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(silu_activation hip::device)

# Add the test executable
add_executable(test_silu test_silu.cpp)
target_link_libraries(test_silu silu_activation hip::device)

# Install targets
install(TARGETS silu_activation DESTINATION lib)
install(FILES SiLU.h DESTINATION include) 