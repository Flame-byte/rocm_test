HIPCC = /usr/bin/hipcc
ROCM_DIR = /opt/rocm

INCLUDES = -I$(ROCM_DIR)/include -I$(ROCM_DIR)/include/hip -I$(ROCM_DIR)/include/hip/amd_detail
CFLAGS = -O2 -std=c++17 -D__HIP_PLATFORM_AMD__

TARGET = fp8
SRC = fp8.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CFLAGS) $(INCLUDES) $^ -o $@

clean:
	rm -f $(TARGET)