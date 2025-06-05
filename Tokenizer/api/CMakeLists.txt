cmake_minimum_required(VERSION 3.10)
project(tokenizer_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Determine the correct library file extension based on platform
if(WIN32)
    set(LIB_PREFIX "")
    if(MSVC)
        set(LIB_SUFFIX ".lib")
    else()
        set(LIB_SUFFIX ".dll.a")
    endif()
elseif(APPLE)
    set(LIB_PREFIX "lib")
    set(LIB_SUFFIX ".dylib")
else() # Linux and others
    set(LIB_PREFIX "lib")
    set(LIB_SUFFIX ".so")
endif()

# Define multiple possible locations for the library
set(POSSIBLE_LIB_PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/target/release/${LIB_PREFIX}tokenizer${LIB_SUFFIX}"
    "${CMAKE_CURRENT_SOURCE_DIR}/target/debug/${LIB_PREFIX}tokenizer${LIB_SUFFIX}"
    "${CMAKE_CURRENT_BINARY_DIR}/${LIB_PREFIX}tokenizer${LIB_SUFFIX}"
)

# Find the first existing library file
foreach(LIB_PATH ${POSSIBLE_LIB_PATHS})
    if(EXISTS ${LIB_PATH})
        set(TOKENIZER_LIB_PATH ${LIB_PATH})
        break()
    endif()
endforeach()

# Add the executable
add_executable(tokenizer_example example.cpp)

# Include directories
target_include_directories(tokenizer_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Custom command to build Rust library first
add_custom_target(build_rust
    COMMAND cargo build --release --features cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Rust library with cargo"
    VERBATIM
)

# Make sure to build Rust library before building the C++ example
add_dependencies(tokenizer_example build_rust)

# If library was found, link against it
if(DEFINED TOKENIZER_LIB_PATH)
    message(STATUS "Found tokenizer library: ${TOKENIZER_LIB_PATH}")
    target_link_libraries(tokenizer_example PRIVATE ${TOKENIZER_LIB_PATH})
else()
    # Otherwise, inform the user what's going on
    message(WARNING "Tokenizer library not found! Building Rust library first...")
    # Create a post-build command to link after Rust build
    add_custom_command(TARGET tokenizer_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Rust library should now be built - you may need to reconfigure CMake"
    )
endif()

# Add install rules
install(TARGETS tokenizer_example DESTINATION bin)
if(DEFINED TOKENIZER_LIB_PATH)
    install(FILES ${TOKENIZER_LIB_PATH} DESTINATION lib)
endif()
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer.h DESTINATION include) 