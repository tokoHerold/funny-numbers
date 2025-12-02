#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <type_traits>

#include "cuda_posit_math.cuh"
#include "posit_gemv.h"

// --- Helper for packed 4-bit atomic updates ---
__device__ void atomic_update_nibble(uint8_t* base_addr, int index, uint8_t val) {
	uintptr_t addr = (uintptr_t)(base_addr + (index / 2));
	uint32_t* aligned_addr = (uint32_t*)(addr & ~3);
	int byte_offset = addr & 3;  // 0, 1, 2, or 3

	bool is_upper = (index & 1);

	// Optimistic synchronisation: CAS
	uint32_t assumed, old = *aligned_addr;
	do {
		assumed = old;

		// Construct new word
		uint32_t mask = 0xFF << (byte_offset * 8);
		uint32_t current_byte = (assumed >> (byte_offset * 8)) & 0xFF;

		uint8_t new_byte;
		if (is_upper) {
			new_byte = (current_byte & 0x0F) | (val << 4);
		} else {
			new_byte = (current_byte & 0xF0) | (val & 0x0F);
		}

		uint32_t new_word = (assumed & ~mask) | (uint32_t(new_byte) << (byte_offset * 8));
		old = atomicCAS(aligned_addr, assumed, new_word);

	} while (assumed != old);  // Repeat on conflict
}

// --- Kernel Definitions ---

template <int N, typename T>
__global__ void k_posit_gemv(const T* A, const T* x, T* y, int dim) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= dim) return;

	T sum = y[row];
	for (int col = 0; col < dim; ++col) {
		T a_val = A[row * dim + col];
		T x_val = x[col];
		T prod = gpu_posit_mul<N, T>(a_val, x_val);
		sum = gpu_posit_add<N, T>(sum, prod);
	}
	y[row] = sum;
}

// Specialization for Posit4 (Packed)
template <>
__global__ void k_posit_gemv<4, uint8_t>(const uint8_t* A, const uint8_t* x, uint8_t* y, int dim) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= dim) return;

	// Read initial Y
	int y_byte_idx = row / 2;
	uint8_t y_raw = y[y_byte_idx];
	uint8_t sum = (row & 1) ? (y_raw >> 4) : (y_raw & 0x0F);

	for (int col = 0; col < dim; ++col) {
		// Read A (Packed)
		int a_idx = row * dim + col;
		uint8_t a_raw = A[a_idx / 2];
		uint8_t a_val = (a_idx & 1) ? (a_raw >> 4) : (a_raw & 0x0F);

		// Read X (Packed)
		uint8_t x_raw = x[col / 2];
		uint8_t x_val = (col & 1) ? (x_raw >> 4) : (x_raw & 0x0F);

		uint8_t prod = gpu_posit_mul<4, uint8_t>(a_val, x_val);
		sum = gpu_posit_add<4, uint8_t>(sum, prod);
	}

	// Write back atomically (because neighbors share a byte)
	atomic_update_nibble(y, row, sum);
}

template <typename T>
__global__ void k_standard_gemv(const T* A, const T* x, T* y, int dim) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= dim) return;

	T sum = 0.0;
	for (int col = 0; col < dim; ++col) {
		sum += A[row * dim + col] * x[col];
	}
	y[row] += sum;
}

// --- Host Launcher ---

template <int N, typename PArray>
void launch_gemv(const PArray& A_host, PArray& x_host, PArray& y_host, int n, int iter) {
	using T = typename StorageType<N>::type;

	// Calculate sizes
	size_t vec_size_bytes, mat_size_bytes;
	if constexpr (N == 4) {
		vec_size_bytes = (n + 1) / 2;
		mat_size_bytes = (n * n + 1) / 2;
	} else {
		vec_size_bytes = n * sizeof(T);
		mat_size_bytes = n * n * sizeof(T);
	}

	T *d_A, *d_x, *d_y;
	CUDA_CHECK(cudaMalloc(&d_A, mat_size_bytes));
	CUDA_CHECK(cudaMalloc(&d_x, vec_size_bytes));
	CUDA_CHECK(cudaMalloc(&d_y, vec_size_bytes));

	// Initial Copy
	std::string name = "All - Posit" + std::to_string(N);
	nvtxRangePush(name.c_str());
	name = "Transfer to - Posit" + std::to_string(N);
	nvtxRangePush(name.c_str());
	CUDA_CHECK(cudaMemcpy(d_A, (void*)A_host.payload.data(), mat_size_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_x, (void*)x_host.data(), vec_size_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_y, (void*)y_host.data(), vec_size_bytes, cudaMemcpyHostToDevice));
	nvtxRangePop();

	int blockSize = 256;
	int gridSize = (n + blockSize - 1) / blockSize;

	for (int i = 0; i < iter; ++i) {
		k_posit_gemv<N, T><<<gridSize, blockSize>>>(d_A, d_x, d_y, n);
		cudaDeviceSynchronize();

		std::swap(d_x, d_y);  // Swap pointers: result y becomes input x for next round
	}

	// Copy back
	name = "Transfer from - Posit" + std::to_string(N);
	nvtxRangePush(name.c_str());
	CUDA_CHECK(cudaMemcpy((void*)x_host.data(), d_x, vec_size_bytes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy((void*)y_host.data(), d_y, vec_size_bytes, cudaMemcpyDeviceToHost));
	nvtxRangePop();
	nvtxRangePop();

	cudaFree(d_A);
	cudaFree(d_x);
	cudaFree(d_y);
}

template <typename T>
void Simulation::ieee754_gemv(const std::vector<T>& A, std::vector<T>& x, std::vector<T>& y) {
	int n = x.size();
	T *d_A, *d_x, *d_y;
	size_t vec_bytes = n * sizeof(T);
	size_t mat_bytes = n * n * sizeof(T);

	CUDA_CHECK(cudaMalloc(&d_A, mat_bytes));
	CUDA_CHECK(cudaMalloc(&d_x, vec_bytes));
	CUDA_CHECK(cudaMalloc(&d_y, vec_bytes));

	std::string name = "All - FP" + std::to_string(sizeof(T) * 8);
	nvtxRangePush(name.c_str());
	name = "Transfer to - FP" + std::to_string(sizeof(T) * 8);
	nvtxRangePush(name.c_str());
	CUDA_CHECK(cudaMemcpy(d_A, A.data(), mat_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_x, x.data(), vec_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_y, y.data(), vec_bytes, cudaMemcpyHostToDevice));
	nvtxRangePop();

	int blockSize = 256;
	int gridSize = (n + blockSize - 1) / blockSize;

	for (int i = 0; i < iterations; ++i) {
		k_standard_gemv<T><<<gridSize, blockSize>>>(d_A, d_x, d_y, n);
		CUDA_CHECK(cudaDeviceSynchronize());
		std::swap(d_x, d_y);  // Swap pointers: result y becomes input x for next round
	}

	name = "Transfer from - FP" + std::to_string(sizeof(T) * 8);
	nvtxRangePush(name.c_str());
	CUDA_CHECK(cudaMemcpy(x.data(), d_x, vec_bytes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(y.data(), d_y, vec_bytes, cudaMemcpyDeviceToHost));
	nvtxRangePop();
	nvtxRangePop();

	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_x));
	CUDA_CHECK(cudaFree(d_y));
}

// --- Implementation of Simulation Wrapper ---
// This is the function called by Simulation class
template <typename PArray>
void Simulation::posit_gemv(const PArray& A, PArray& x, PArray& y) {
	// Dispatch based on PArray type
	int n = N, iter = iterations;
	if constexpr (std::is_same_v<PArray, Posit64Array>)
		launch_gemv<64>(A, x, y, n, iter);
	else if constexpr (std::is_same_v<PArray, Posit32Array>)
		launch_gemv<32>(A, x, y, n, iter);
	else if constexpr (std::is_same_v<PArray, Posit16Array>)
		launch_gemv<16>(A, x, y, n, iter);
	else if constexpr (std::is_same_v<PArray, Posit8Array>)
		launch_gemv<8>(A, x, y, n, iter);
	else if constexpr (std::is_same_v<PArray, Posit4Array>)
		launch_gemv<4>(A, x, y, n, iter);
}

int main(int argc, char** argv) {
	auto [n, iter, rows] = Simulation::parse_args(argc, argv);
	std::cout << "Running GPU accuracy test for N=" << n << " and I=" << iter << std::endl;
	Simulation simulation{n, iter};
	std::cout << "Conversion inaccuracy:" << std::endl;
	simulation.report_accuracy(true);
	simulation.run();
	if (rows) {
		std::cout << "First " << rows << " results:" << std::endl;
		std::cout << "Double\tPosit64\tPosit32\tPosit16\tPosit8\tPosit4" << std::endl;
	}
	for (size_t row = 0; row < rows; row++) {
		std::cout << simulation.x_f64[row] << '\t' << simulation.x_p64[row].to_double() << '\t';
		std::cout << simulation.x_p32[row].to_double() << '\t';
		std::cout << simulation.x_p16[row].to_double() << '\t';
		std::cout << simulation.x_p08[row].to_double() << '\t';
		std::cout << simulation.x_p04[row].to_double() << '\t' << std::endl;
	}
	simulation.report_accuracy(true);
}
