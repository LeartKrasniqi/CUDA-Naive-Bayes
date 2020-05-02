# CUDA Flags
CC = aarch64-linux-gnu-g++
NVCC = /usr/local/cuda-10.0/bin/nvcc -ccbin $(CC)
NVCC_LIB = /usr/local/cuda-10.0/targets/aarch64-linux/lib
INCLUDES = /usr/local/cuda-10.0/samples/common/inc
CUDART_LIB = /usr/local/cuda-10.0/targets/aarch64-linux/lib/libcudart.so

# CPU Flags
CXX = g++
CXXFLAGS = -std=c++0x -fopenmp
CFLAGS = -Wall -Wextra

nb_gpu: nb_gpu.o
	$(CC) $(CXXFLAGS) -o nb_gpu nb_gpu.o -L$(NVCC_LIB) -lcudart

nb_gpu.o: ./nb_gpu.cu
	$(NVCC) -I. -I$(INCLUDES) -c ./nb_gpu.cu

nb_cpu.out: nb_cpu.o nb.o
	$(CXX) -o nb_cpu.out nb_cpu.o nb.o $(CXXFLAGS)

nb_cpu.o: nb_cpu.cpp
	$(CXX) -c nb_cpu.cpp $(CXXFLAGS)

nb.o: ./include/nb.cpp ./include/nb.hpp
	$(CXX) -c ./include/nb.cpp $(CXXFLAGS)

clean:
	rm -rf *.exe *.o *.stackdump *~
