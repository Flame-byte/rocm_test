# Makefile for ROCm/HIP MoE project

HIPCC := hipcc
-CXXFLAGS := -std=c++14 -O2 -IMoE
+CXXFLAGS := -std=c++14 -O2 -fPIC -IMoE
LDFLAGS := -lrocblas

# Library target
LIB := MoE
LIB_SRCS := \
    MOE.cpp \
    gating_network.cpp \
    expert.cpp \
    router.cpp \
    linear.cpp \
    dot_product.cpp \
    softmax.cpp \
    SiLU.cpp
LIB_OBJS := $(LIB_SRCS:.cpp=.o)

# Inference executable
EXE := moe_infer
EXE_SRCS := MoE/moe_infer.cpp
EXE_OBJS := $(EXE_SRCS:.cpp=.o)

.PHONY: all clean
all: $(LIB) $(EXE)

# Build shared library
$(LIB): $(LIB_OBJS)
	$(HIPCC) -shared -o $@ $^ $(LDFLAGS)

# Build inference executable
$(EXE): $(EXE_OBJS) $(LIB)
	$(HIPCC) -o $@ $^ -L. -lmoe $(LDFLAGS)

# Generic rule for object files
%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(LIB_OBJS) $(EXE_OBJS) $(LIB) $(EXE) 