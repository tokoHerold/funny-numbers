#pragma once

#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "packed_posit.h"

/**
 * @brief Class that manages the GEMV simulation data and execution flow.
 * * This class handles:
 * - Data Initialization (random generation of Matrix A and Vector V).
 * - Ground Truth calculation (using double precision).
 * - Accuracy Reporting (MSE calculation).
 */
class Simulation {
	void print_usage(const char* prog_name);  // forward declaration
	void construct(int argc, char** argv);    // forward declaration

   public:
	size_t N;
	int iterations;

	// A is the matrix (row-major), V is the vector.
	std::vector<double> A_f64, V_f64;
	Posit64Array A_p64, V_p64;
	Posit32Array A_p32, V_p32;
	Posit16Array A_p16, V_p16;
	Posit8Array A_p08, V_p08;
	Posit4Array A_p04, V_p04;

	/**
	 * @brief Construct a new Simulation object.
	 *
	 * @param n Size of the vector / matrix dimension (NxN matrix)
	 * @param iter Number of iterations to run V += A*V
	 * @param seed Random seed for initialization
	 */
	Simulation(size_t n, int iter, int seed = 0xbf2)
	    : N(n),
	      iterations(iter),
	      A_p64(n * n),
	      V_p64(n),
	      A_p32(n * n),
	      V_p32(n),
	      A_p16(n * n),
	      V_p16(n),
	      A_p08(n * n),
	      V_p08(n),
	      A_p04(n * n),
	      V_p04(n) {
		if (n == 0) throw std::invalid_argument("n must not be zero!");
		initialize_data(seed);
	}

	/**
	 * @brief Construct a new Simulation object from command line arguments.
	 */
	Simulation(int argc, char** argv)
	    : N(0),
	      iterations(0),  // placeholders, overwritten by construct
	      A_p64(0),
	      V_p64(0),
	      A_p32(0),
	      V_p32(0),
	      A_p16(0),
	      V_p16(0),
	      A_p08(0),
	      V_p08(0),
	      A_p04(0),
	      V_p04(0) {
		construct(argc, argv);
	}

	/**
	 * @brief Initializes arrays with random data.
	 *
	 * * @param seed Random seed.
	 */
	void initialize_data(int seed) {
		std::mt19937 gen(seed);
		std::uniform_real_distribution<> dis(0.0, 1.0);

		A_f64.resize(N * N);
		V_f64.resize(N);

		// Initialize Matrix A
		for (size_t i = 0; i < N * N; ++i) {
			double val = dis(gen);

			A_f64[i] = val;
			A_p64[i] = Posit64(val);
			A_p32[i] = Posit32(val);
			A_p16[i] = Posit16(val);
			A_p08[i] = Posit8(val);
			A_p04[i] = Posit4(val);
		}

		// Initialize Vector V
		for (size_t i = 0; i < N; ++i) {
			double val = dis(gen);

			V_f64[i] = val;
			V_p64[i] = Posit64(val);
			V_p32[i] = Posit32(val);
			V_p16[i] = Posit16(val);
			V_p08[i] = Posit8(val);
			V_p04[i] = Posit4(val);
		}
	}

	/**
	 * @brief This function executes the GEMV loops and reports accuracy.
	 */
	void run() {
		double_gemv(A_f64, V_f64);
		posit_gemv(A_p04, V_p04);
		posit_gemv(A_p08, V_p08);
		posit_gemv(A_p16, V_p16);
		posit_gemv(A_p32, V_p32);
		posit_gemv(A_p64, V_p64);
		report_accuracy();
	}

   private:
	/*
	 * @brief Executes the specified number of GEMV operations on the Posit data type
	 */
	template <typename PArray>
	void posit_gemv(const PArray& A, PArray& V);

	/*
	 * @brief Executes the specified number of GEMV operations on the double data type
	 */
	void double_gemv(const std::vector<double>& A, std::vector<double>& V);

	/**
	 * @brief Calculates Mean Squared Error of Relative Error against V_double.
	 */
	template <typename PArray>
	double calculate_mse(const PArray& V_p) {
		double sum_sq_error = 0.0;
		for (size_t i = 0; i < N; ++i) {
			double oracle = V_f64[i];
			double val = V_p[i].to_double();

			double err = 0.0;
			if (std::abs(oracle) > 1e-10) {  // Prevent division-by-zero
				err = std::abs(val - oracle) / std::abs(oracle);
			} else {
				err = std::abs(val - oracle);
			}
			sum_sq_error += err * err;
		}
		return sum_sq_error / N;
	}

	void print_metric(const std::string& name, double mse) {
		std::cout << std::setw(10) << name << std::setw(20) << std::scientific << mse << std::endl;
	}

	/**
	 * @brief Reports accuracy metrics to stdout.
	 */
	void report_accuracy() {
		std::cout << "\n--- Accuracy Report (MSE of Relative Error) ---\n";
		std::cout << std::setw(10) << "Type" << std::setw(20) << "MSE" << std::endl;
		std::cout << std::string(30, '-') << std::endl;

		print_metric("Posit64", calculate_mse(V_p64));
		print_metric("Posit32", calculate_mse(V_p32));
		print_metric("Posit16", calculate_mse(V_p16));
		print_metric("Posit8", calculate_mse(V_p08));
		print_metric("Posit4", calculate_mse(V_p04));
	}
};

inline void Simulation::print_usage(const char* prog_name) {
	std::cerr << "Usage: " << prog_name << " -N <size> [-I <iterations>]\n"
	          << "  -N <size>:       Number of elements (vector size/matrix dim)\n"
	          << "  -I <iterations>: Number of GEMV iterations (default: 1)\n";
}

inline void Simulation::construct(int argc, char** argv) {
	int N = 0;
	int iterations = 1;
	int opt;

	// Parse command-line arguments
	while ((opt = getopt(argc, argv, "N:I:")) != -1) {
		switch (opt) {
			case 'N':
				try {
					N = std::stoi(optarg);
				} catch (...) {
					std::cerr << "Error: Invalid size provided for -N.\n";
					print_usage(argv[0]);
					std::exit(EXIT_FAILURE);
				}
				break;
			case 'I':
				try {
					iterations = std::stoi(optarg);
				} catch (...) {
					std::cerr << "Error: Invalid iterations provided for -I.\n";
					print_usage(argv[0]);
					std::exit(EXIT_FAILURE);
				}
				break;
			default: /* '?' */
				print_usage(argv[0]);
				std::exit(EXIT_FAILURE);
		}
	}

	// Validate Arguments
	if (N <= 0) {
		std::cerr << "Error: Size N must be greater than 0.\n";
		print_usage(argv[0]);
		std::exit(EXIT_FAILURE);
	}

	if (iterations <= 0) {
		std::cerr << "Error: Iterations must be greater than 0.\n";
		print_usage(argv[0]);
		std::exit(EXIT_FAILURE);
	}
	new (this) Simulation(N, iterations);
}
