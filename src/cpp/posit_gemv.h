#pragma once

#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
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
	size_t N;
	int iterations;

	/*
	 * @brief Executes the specified number of GEMV operations on the Posit data type
	 */
	template <typename PArray>
	void posit_gemv(const PArray& A, PArray& x, PArray& y);

	/*
	 * @brief Executes the specified number of GEMV operations on the double data type
	 */
	template<typename T>
	void ieee754_gemv(const std::vector<T>& A, std::vector<T>& x, std::vector<T>& y);

	/*
	 * @brief Executes the specified number of GEMV operations on the float data type
	 */
	void float_gemv(const std::vector<float>& A, std::vector<float>& x, std::vector<float>& y);

	/**
	 * @brief Calculates Mean Squared Error of Relative Error against V_double.
	 */
	template <typename PArray>
	double calculate_mse(const PArray& V_p, const std::vector<double>& oracle) {
		double sum_sq_error = 0.0;
		for (size_t i = 0; i < N; ++i) {
			double val = V_p[i].to_double();
			double truth = oracle[i];

			double err = 0.0;
			if (std::abs(truth) > 1e-10) {  // Prevent division-by-zero
				err = std::abs(val - truth) / std::abs(truth);
			} else {
				err = std::abs(val - truth);
			}
			sum_sq_error += err * err;
		}
		return sum_sq_error / N;
	}

	/**
	 * @brief Calculates Mean Squared Error for a float vector against V_double.
	 * Overload for std::vector<float>.
	 */
	double calculate_mse(const std::vector<float>& V_f, const std::vector<double>& oracle) {
		double sum_sq_error = 0.0;
		for (size_t i = 0; i < N; ++i) {
			double val = static_cast<double>(V_f[i]);
			double truth = oracle[i];

			double err = 0.0;
			if (std::abs(truth) > 1e-10) {
				err = std::abs(val - truth) / std::abs(truth);
			} else {
				err = std::abs(val - truth);
			}
			sum_sq_error += err * err;
		}
		return sum_sq_error / N;
	}

	void print_metric(const std::string& name, double mse) {
		std::cout << std::setw(10) << name << std::setw(20) << std::scientific << mse << std::endl;
	}

   public:
	// A is the matrix (row-major), V is the vector.
	std::vector<double> A_f64, x_f64, y_f64;
	std::vector<float> A_f32, x_f32, y_f32;
	Posit64Array A_p64, x_p64, y_p64;
	Posit32Array A_p32, x_p32, y_p32;
	Posit16Array A_p16, x_p16, y_p16;
	Posit8Array A_p08, x_p08, y_p08;
	Posit4Array A_p04, x_p04, y_p04;

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
			A_p64(n * n), x_p64(n), y_p64(n),
			A_p32(n * n), x_p32(n), y_p32(n),
			A_p16(n * n), x_p16(n), y_p16(n),
			A_p08(n * n), x_p08(n), y_p08(n),
			A_p04(n * n), x_p04(n), y_p04(n) {
		if (n == 0) throw std::invalid_argument("n must not be zero!");
		initialize_data(seed);
	}

	/**
	 * @brief Initializes arrays with random data.
	 *
	 * * @param seed Random seed.
	 */
	void initialize_data(int seed) {
		std::mt19937 gen(seed);
		std::uniform_real_distribution<> dis(-0.1, 0.1);

		A_f64.resize(N * N);
		x_f64.resize(N);
		y_f64.resize(N);
		A_f32.resize(N * N);
		x_f32.resize(N);
		y_f32.resize(N);

		// Initialize Matrix A
		for (size_t i = 0; i < N * N; ++i) {
			double val = dis(gen);

			A_f32[i] = A_f64[i] = val;
			A_p64[i] = Posit64(val);
			A_p32[i] = Posit32(val);
			A_p16[i] = Posit16(val);
			A_p08[i] = Posit8(val);
			A_p04[i] = Posit4(val);
		}

		// Initialize Vector V
		for (size_t i = 0; i < N; ++i) {
			double val = dis(gen);

			y_f32[i] = x_f32[i] = y_f64[i] = x_f64[i] = val;
			y_p64[i] = x_p64[i] = Posit64(val);
			y_p32[i] = x_p32[i] = Posit32(val);
			y_p16[i] = x_p16[i] = Posit16(val);
			y_p08[i] = x_p08[i] = Posit8(val);
			y_p04[i] = x_p04[i] = Posit4(val);
		}
	}

	/**
	 * @brief This function executes the GEMV loops.
	 */
	void run() {
		ieee754_gemv(A_f64, x_f64, y_f64);
		ieee754_gemv(A_f32, x_f32, y_f32);
		posit_gemv(A_p04, x_p04, y_p04);
		posit_gemv(A_p08, x_p08, y_p08);
		posit_gemv(A_p16, x_p16, y_p16);
		posit_gemv(A_p32, x_p32, y_p32);
		posit_gemv(A_p64, x_p64, y_p64);
	}

	/**
	 * @brief Reports accuracy metrics to stdout.
	 */
	void report_accuracy(bool check_x = false) {
		const auto& ref = check_x ? x_f64 : y_f64;

		std::string target = check_x ? "X" : "Y";
		std::cout << "\n--- Accuracy Report (" << target << ") ---\n";
		std::cout << std::setw(10) << "Type" << std::setw(20) << "MSE" << std::endl;
		std::cout << std::string(30, '-') << std::endl;

		if (check_x) {
			print_metric("Float32", calculate_mse(x_f32, ref));
			print_metric("Posit64", calculate_mse(x_p64, ref));
			print_metric("Posit32", calculate_mse(x_p32, ref));
			print_metric("Posit16", calculate_mse(x_p16, ref));
			print_metric("Posit8", calculate_mse(x_p08, ref));
			print_metric("Posit4", calculate_mse(x_p04, ref));
		} else {
			print_metric("Float32", calculate_mse(y_f32, ref));
			print_metric("Posit64", calculate_mse(y_p64, ref));
			print_metric("Posit32", calculate_mse(y_p32, ref));
			print_metric("Posit16", calculate_mse(y_p16, ref));
			print_metric("Posit8", calculate_mse(y_p08, ref));
			print_metric("Posit4", calculate_mse(y_p04, ref));
		}
	}

	static void print_usage(const char* prog_name) {
		std::cerr << "Usage: " << prog_name << " -N <size> [-I <iterations> -V <rows>]\n"
		          << "  -N <size>:       Number of elements (vector size/matrix dim)\n"
		          << "  -I <iterations>: Number of GEMV iterations (default: 1)\n"
		          << "  -V <rows>:       Number of rows to print of arrays (default: 0)\n";
	}

	static std::tuple<size_t, int, size_t> parse_args(int argc, char** argv) {
		size_t N = 0;
		int iterations = 1;
		size_t rows = 0;
		int opt;

		// Parse command-line arguments
		while ((opt = getopt(argc, argv, "N:I:V:")) != -1) {
			switch (opt) {
				case 'N':
					try {
						N = std::stoul(optarg);
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
				case 'V':
					try {
						rows = std::stoul(optarg);
					} catch (...) {
						std::cerr << "Error: Invalid size provided for -V.\n";
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
		return {N, iterations, rows};
	}
};
