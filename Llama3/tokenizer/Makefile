CXX = hipcc
CXXFLAGS = -std=c++14 -g -O2

# Add the path to the Rust library
RUST_LIB_PATH = ../api/target/release
LIBS = -L$(RUST_LIB_PATH) -ltokenizer -Wl,-rpath,$(RUST_LIB_PATH)

TARGET = tokenizer_test
SRCS = tokenizer.cpp load.cpp
OBJS = $(SRCS:.cpp=.o)
HEADERS = tokenizer.h load.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) 