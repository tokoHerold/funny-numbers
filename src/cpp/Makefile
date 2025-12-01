CXX      = g++
NVCC     = nvcc

CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -fopenmp
NVCCFLAGS= -std=c++20 -O3 -arch=native -Xcompiler "-fopenmp"

# --- Libraries ---
# Link against BLAS. 
# On many systems this is -lblas or -lopenblas. 
# On HPC systems with modules loaded, this might sometimes be empty or specific (e.g. -sci_lib).
# LIBS     = -lopenblas
# CUDA_LIBS = -lcublas
CUDA_LIBS = -L$(NVIDIA_PATH)/math_libs/lib64 -lcublas -lcublasLt -lcudart

# --- Targets ---
all: posit_test cpu_posit_gemv cuda_posit_gemv

posit.o: posit.cc posit.h packed_posit.h
	$(CXX) $(CXXFLAGS) -c posit.cc -o posit.o

# ==========================================
# Target 1: Posit Test
# ==========================================
posit_test: posit_test.cc posit.o
	$(CXX) $(CXXFLAGS) posit_test.cc posit.o -o posit_test

# ==========================================
# Target 2: CPU GEMV (Baseline)
# ==========================================
cpu_posit_gemv.o: cpu_posit_gemv.cc posit_gemv.h
	$(CXX) $(CXXFLAGS) -c cpu_posit_gemv.cc -o cpu_posit_gemv.o $(LIBS)

cpu_posit_gemv: cpu_posit_gemv.o posit.o
	$(CXX) $(CXXFLAGS) cpu_posit_gemv.o posit.o -o cpu_posit_dgemv $(LIBS)

# ==========================================
# Target 3: CUDA GEMV
# ==========================================
cuda_posit_gemv.o: cuda_posit_gemv.cu cuda_posit_math.cuh posit_gemv.h
	$(NVCC) $(NVCCFLAGS) -c cuda_posit_gemv.cu -o cuda_posit_gemv.o $(CUDA_LIBS)

cuda_posit_gemv:  cuda_posit_gemv.o posit.o
	$(NVCC) $(NVCCFLAGS) cuda_posit_gemv.o posit.o -o cuda_posit_dgemv $(CUDA_LIBS)

# ==========================================
# Utilities
# ==========================================
clean:
	rm -f *.o posit_test cpu_posit_dgemv cuda_posit_dgemv
