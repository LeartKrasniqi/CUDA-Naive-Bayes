# CUDA Flags 
CC = aarch64-linux-gnu-g++
NVCC = /usr/local/cuda-10.0/bin/nvcc -ccbin $(CC)
NVCC_LIB = /usr/local/cuda-10.0/targets/aarch64-linux/lib
INCLUDES = /usr/local/cuda-10.0/samples/common/inc
CUDART_LIB = /usr/local/cuda-10.0/targets/aarch64-linux/lib/libcudart.so

# CPU Flags
CXX = g++
CXXFLAGS = -std=c++0x
CFLAGS = -Wall -Wextra


nb_cpu.out: nb_cpu.o nb.o
	$(CXX) -o nb_cpu.out nb_cpu.o nb.o  

nb_cpu.o: nb_cpu.cpp 
	$(CXX) -c nb_cpu.cpp $(CXXFLAGS)

nb.o: ./include/nb.cpp ./include/nb.hpp
	$(CXX) -c ./include/nb.cpp

clean:
	rm -rf *.exe *.o *.stackdump *~