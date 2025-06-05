HIPCC = /usr/bin/hipcc
TORCH_DIR = /home/qin/libtorch_rocm/libtorch
ROCM_DIR = /opt/rocm

INCLUDES = -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -I$(ROCM_DIR)/include -I$(ROCM_DIR)/include/hip -I$(ROCM_DIR)/include
LIBS = -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 -ltorch_hip -lc10_hip -lamdhip64
CFLAGS = -O2 -std=c++17 -D__HIP_PLATFORM_AMD__

TARGET = fp8_mm_test
SRC = fp8.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LIBS) -Wl,-rpath,$(TORCH_DIR)/lib

clean:
	rm -f $(TARGET)