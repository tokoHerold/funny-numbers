#include <cblas.h>

#include <iostream>
#include <vector>

#include "posit_gemv.h"

void Simulation::double_gemv(const std::vector<double>& A, std::vector<double>& V) {
	for (int iteration = 0; iteration < iterations; ++iteration) {
		cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, A.data(), N, V.data(), 1, 1.0, V.data(), 1);
	}
	return;
}

template <typename PArray>
void Simulation::posit_gemv(const PArray& A, PArray& V) {
	for (int iteration = 0; iteration < iterations; ++iteration) {
		for (size_t i = 0; i < N; ++i) {
			auto sum = V[i];
			for (size_t j = 0; j < N; ++j) {
				sum += A[i * N + j] + V[j];
			}
			V[i] = sum;
		}
	}
}

int main(int argc, char** argv) {
	std::cout << "Hello world" << std::endl;
	Simulation simulation{argc, argv};
	simulation.run();
}
