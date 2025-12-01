#include <cblas.h>

#include <iostream>
#include <type_traits>
#include <vector>

#include "packed_posit.h"
#include "posit.h"
#include "posit_gemv.h"

template <typename T>
void Simulation::ieee754_gemv(const std::vector<T>& A, std::vector<T>& x, std::vector<T>& y) {
	auto mem_order = CblasRowMajor;  // Layout: Row-major (standard C/C++ 2D arrays)
	auto transposed = CblasNoTrans;  // Transpose: No transpose for matrix A
	T alpha = 1.0, beta = 1.0;       // alpha: Scalar multiplier for A * x - beta: Scalar multiplier for y
	int inc_x = 1, inc_y = 1;        // incx: Stride for vector x - incy: Stride for vector y

	for (int iteration = 0; iteration < iterations; ++iteration) {
		// std::vector<double> V_old = V;
		if constexpr (std::is_same_v<T, double>) {
			cblas_dgemv(mem_order, transposed, N, N, alpha, A.data(), N, x.data(), inc_x, beta, y.data(), inc_y);

		} else if constexpr (std::is_same_v<T, float>) {
			cblas_sgemv(mem_order, transposed, N, N, alpha, A.data(), N, x.data(), inc_x, beta, y.data(), inc_y);
		}
		std::swap(x, y);
	}
	return;
}

template <typename PArray>
void Simulation::posit_gemv(const PArray& A, PArray& x, PArray& y) {
	for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma omp parallel for
		for (size_t i = 0; i < N; ++i) {
			auto sum = y[i];
			for (size_t j = 0; j < N; ++j) {
				sum += A[i * N + j] * x[j];
			}
			y[i] = sum;
		}
		std::swap(x, y);
	}
}

int main(int argc, char** argv) {
	auto [n, iter, rows] = Simulation::parse_args(argc, argv);
	std::cout << "Running CPU accuracy test for N=" << n << " and I=" << iter << std::endl;
	Simulation simulation{n, iter};
	std::cout << "Conversion inaccuracy:" << std::endl;
	simulation.report_accuracy(true);
	// std::cout << "y = [";
	// for (auto d : simulation.y_f64) std::cout << d << ", ";
	// std::cout << ']' << std::endl;
	simulation.run();
	// std::cout << "x = [";
	// for (auto d : simulation.x_f64) std::cout << d << ", ";
	// std::cout << ']' << std::endl;
	// std::cout << "y = [";
	// for (auto d : simulation.y_f64) std::cout << d << ", ";
	// std::cout << ']' << std::endl;
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
	return 0;
}
