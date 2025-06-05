# Tokenizer C++ Bindings

This project provides C++ bindings for the Rust tokenizer implementation. It allows you to use the efficient Rust-based BPE tokenizer in your C++ applications.

## Building the Library

### Prerequisites

- Rust (1.60+ recommended)
- Cargo (comes with Rust)
- CMake (3.10+)
- A C++20 compatible compiler (GCC 10+, Clang 10+, or MSVC 2019+)

### Build Steps

1. Build the Rust library with the `cpp` feature:

```bash
cd Tokenizer
cargo build --release --features cpp
```

2. Build the C++ example using CMake:

```bash
mkdir build && cd build
cmake ..
make
```

3. Run the example:

```bash
./tokenizer_example
```

## Using the Library in Your C++ Project

### Option 1: CMake Integration

Add the following to your CMakeLists.txt:

```cmake
# Path to the Tokenizer directory
set(TOKENIZER_PATH "/path/to/Tokenizer")

# Add the tokenizer library
add_subdirectory(${TOKENIZER_PATH} tokenizer)

# Link your target with the tokenizer
target_link_libraries(your_target PRIVATE tokenizer)
```

### Option 2: Manual Integration

1. Build the Rust library:

```bash
cd /path/to/Tokenizer
cargo build --release --features cpp
```

2. Include the header file in your C++ code:

```cpp
#include "tokenizer.h"
```

3. Link with the generated library:
   - On Linux/macOS: `libtokenizer.so` or `libtokenizer.a`
   - On Windows: `tokenizer.dll` or `tokenizer.lib`

## API Usage

Here's a simple example of how to use the tokenizer in your C++ code:

```cpp
#include "tokenizer.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    try {
        // Create a simple tokenizer
        std::vector<std::pair<std::vector<uint8_t>, Rank>> encoder = {
            {{'h', 'e', 'l', 'l', 'o'}, 0},
            {{' ', 'w', 'o', 'r', 'l', 'd'}, 1}
        };
        
        std::vector<std::pair<std::string, Rank>> special_tokens = {
            {"<|endoftext|>", 2}
        };
        
        // Use standard GPT-2 pattern
        std::string pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
        
        // Create tokenizer
        Tokenizer tokenizer(encoder, special_tokens, pattern);
        
        // Encode text
        std::string text = "hello world";
        std::vector<Rank> tokens = tokenizer.EncodeOrdinary(text);
        
        // Print tokens
        for (Rank token : tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
        
        // Decode tokens
        std::string decoded = tokenizer.Decode(tokens);
        std::cout << "Decoded: " << decoded << std::endl;
        
    } catch (const TokenizerException& e) {
        std::cerr << "Tokenizer error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## API Reference

### Tokenizer Class

The main class for tokenizing text:

```cpp
class Tokenizer {
public:
    // Constructor
    Tokenizer(
        const std::vector<std::pair<std::vector<uint8_t>, Rank>>& encoder,
        const std::vector<std::pair<std::string, Rank>>& special_tokens,
        const std::string& pattern
    );
    
    // Destructor
    ~Tokenizer();
    
    // Encode ordinary text (ignoring special tokens)
    std::vector<Rank> EncodeOrdinary(const std::string& text);
    
    // Encode text with special tokens
    std::vector<Rank> Encode(
        const std::string& text,
        const std::vector<std::string>& allowed_special = {}
    );
    
    // Decode tokens to string
    std::string Decode(const std::vector<Rank>& tokens);
    
    // Encode a single token
    Rank EncodeSingleToken(const std::vector<uint8_t>& bytes);
    Rank EncodeSingleToken(const std::string& text);
    
    // Get all special tokens
    std::vector<std::pair<std::string, Rank>> GetSpecialTokens();
};
```

## Error Handling

The library uses exceptions for error handling:

```cpp
// Catch tokenizer exceptions
try {
    // Tokenizer code here
} catch (const TokenizerException& e) {
    std::cerr << "Tokenizer error: " << e.what() 
              << " (error code: " << static_cast<int>(e.error_code) << ")" << std::endl;
}
```

## Thread Safety

The Tokenizer class is not thread-safe. If you need to use the tokenizer from multiple threads:

1. Create separate Tokenizer instances for each thread, or
2. Protect access to a shared Tokenizer with a mutex 