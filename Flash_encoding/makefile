# Use hipcc for compiling HIP code
HIPCC = hipcc

# Target executable name
TARGET = flash_attention_test

# Source files
SRCS = test_flash_mask.cpp flash_mask.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Compiler flags
# -I. adds the current directory to the include path, needed for flash_mask.h
HIPFLAGS = -I. -std=c++20 # Adding -std=c++20 for modern C++ features

# Default target: build the executable
all: $(TARGET)

# Rule to link object files into the executable
$(TARGET): $(OBJS)
	$(HIPCC) $(OBJS) -o $(TARGET)

# Rule to compile C++ source files into object files
%.o: %.cpp
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Clean rule to remove generated files
clean:
	rm -f $(TARGET) $(OBJS)