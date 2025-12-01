#include <cblas.h>

#include <iostream>
#include <vector>

#include "packed_posit.h"
#include "posit.h"
#include "posit_gemv.h"

void Simulation::double_gemv(const std::vector<double>& A, std::vector<double>& x, std::vector<double>& y) {
	for (int iteration = 0; iteration < iterations; ++iteration) {
		// std::vector<double> V_old = V;
		cblas_dgemv(CblasRowMajor,  // Layout: Row-major (standard C/C++ 2D arrays)
		            CblasNoTrans,   // Transpose: No transpose for matrix A
		            N,              // M: Number of rows in matrix A
		            N,              // N: Number of columns in matrix A
		            1.0,            // alpha: Scalar multiplier for A * x
		            A.data(),       // A: Pointer to the matrix A data
		            N,              // lda: Leading dimension of A (stride between rows)
		            x.data(),       // x: Pointer to the vector x data
		            1,              // incx: Stride for vector x
		            1.0,            // beta: Scalar multiplier for y
		            y.data(),       // y: Pointer to the vector y data (output)
		            1);             // incy: Stride for vector y
		std::swap(x, y);
	}
	return;
}

void Simulation::float_gemv(const std::vector<float>& A, std::vector<float>& x, std::vector<float>& y) {
	for (int iteration = 0; iteration < iterations; ++iteration) {
		// std::vector<double> V_old = V;
		cblas_sgemv(CblasRowMajor,  // Layout: Row-major (standard C/C++ 2D arrays)
		            CblasNoTrans,   // Transpose: No transpose for matrix A
		            N,              // M: Number of rows in matrix A
		            N,              // N: Number of columns in matrix A
		            1.0,            // alpha: Scalar multiplier for A * x
		            A.data(),       // A: Pointer to the matrix A data
		            N,              // lda: Leading dimension of A (stride between rows)
		            x.data(),       // x: Pointer to the vector x data
		            1,              // incx: Stride for vector x
		            1.0,            // beta: Scalar multiplier for y
		            y.data(),       // y: Pointer to the vector y data (output)
		            1);             // incy: Stride for vector y
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
	simulation.run();
	if (rows) {
		std::cout << "First " << rows << " results:" << std::endl;
		std::cout << "Double\tPosit64\tPosit32\tPosit16\tPosit8\tPosit4" << std::endl;
	}
	for (size_t row = 0; row < rows; row++) {
		std::cout << simulation.y_f64[row] << '\t' << simulation.x_p64[row].to_double() << '\t';
		std::cout << simulation.y_p32[row].to_double() << '\t';
		std::cout << simulation.y_p16[row].to_double() << '\t';
		std::cout << simulation.y_p08[row].to_double() << '\t';
		std::cout << simulation.y_p04[row].to_double() << '\t' << std::endl;
	}
	simulation.report_accuracy(iter % 2 == 1);
#if false
	std::vector<double> data = {0.68388487739347992, 0.95939352514962573, 0.68712501907613477, 0.72565265983749538,
	                            0.97459122967335732, 0.48655151260357221, 0.8233789504219573,  0.030202757702509214};

	using T = Posit16;
	using S = Posit16Array;
	double result_double = data[0];
	T result_posit(0.0);
	S m{8};
	S v{8};

	for (int i = 0; i < 8; ++i) {
		m[i] = T(data[i]);
		v[i] = T(data[i]);
	}

	std::cout << std::fixed << std::setprecision(16);
	std::cout << "Step | Posit   Value        | Double (Correct) Value | Abs Error\n";
	std::cout << "-----|----------------------|------------------------|------------------\n";

	for (size_t i = 0; i < data.size(); ++i) {
		double val = data[i];

		// Double precision calculation
		double sq_double = val * val;
		result_double += sq_double;

		// Posit64 calculation
		T p_val = m[i];

		// Using operator*= (which uses the code you highlighted)
		// Note: Creating a temporary for square to match "A[0] * A[0]" semantics before accumulation
		T p_sq = p_val;
		p_sq *= p_val;

		// Using operator+= (which uses the code you highlighted)
		v[0] += p_sq;

		// Compare
		double posit_as_double = v[0].to_double();
		double error = std::abs(posit_as_double - result_double);

		std::cout << std::setw(4) << (i + 1) << " | " << posit_as_double << " | " << result_double << " | " << error
		          << "\n";
	}

	std::cout << "\nFinal Results:\n";
	std::cout << "Double : " << result_double << "\n";
	std::cout << "Posit: " << result_posit.to_double() << "\n";
#endif

	return 0;
}
