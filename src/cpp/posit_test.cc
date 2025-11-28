#include "posit.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

static int global_fails = 0;

#define assert_eq(statement, expected)                                                   \
	{                                                                                    \
		auto res = (statement);                                                          \
		if constexpr (std::is_floating_point<decltype(res)>::value) {                    \
			if ((std::isnan(res) && std::isnan(expected)) || (res == expected)) {        \
				/* Values are either equal or both NaN */                                \
			} else {                                                                     \
				std::cerr << " ✗" << std::endl << #statement << " failed; got: " << res; \
				std::cerr << ", expected: " << (expected) << std::endl;                  \
				global_fails++;                                                          \
				return;                                                                  \
			}                                                                            \
		} else {                                                                         \
			if (res != (expected)) {                                                     \
				std::cerr << " ✗" << std::endl << #statement << " failed; got: " << res; \
				std::cerr << ", expected: " << (expected) << std::endl;                  \
				global_fails++;                                                          \
				return;                                                                  \
			}                                                                            \
		}                                                                                \
	}

void test_conversion() {
	double input[] = {0, std::nan(""), 1337, 0.1337, -1, -1.5, -0.5, -0.75, -0.1337, 0.123456789, -0.123456789};
	double results_4[] = {0.0, std::nan(""), 256.0, 0.25, -1.0, -1.0, -1.0, -1.0, -0.25, 0.0625, -0.0625};
	double results_8[] = {0.0, std::nan(""), 1536.0, 0.140625, -1.0, -1.5, -0.5, -0.75, -0.140625, 0.125, -0.125};
	double results_16[] = {0.0,
	                       std::nan(""),
	                       1336.0,
	                       0.13372802734375,
	                       -1.0,
	                       -1.5,
	                       -0.5,
	                       -0.75,
	                       -0.13372802734375,
	                       0.123443603515625,
	                       -0.123443603515625};
	double results_32[] = {0.0,
	                       std::nan(""),
	                       1337.0,
	                       0.13370000012218952,
	                       -1.0,
	                       -1.5,
	                       -0.5,
	                       -0.75,
	                       -0.13370000012218952,
	                       0.1234567891806364,
	                       -0.1234567891806364};
	double results_64[] = {0.0,  std::nan(""), 1337.0,  0.1337,      -1.0,        -1.5,
	                       -0.5, -0.75,        -0.1337, 0.123456789, -0.123456789};

	std::cout << "=== Test double conversion ===" << std::endl;
	for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
		auto val = input[i];
		std::cout << "\tTesting " << val << std::flush;
		auto a4(Posit4{val});
		auto a8(Posit8{val});
		auto a16(Posit16{val});
		auto a32(Posit32{val});
		auto a64(Posit64{val});

		assert_eq((a4).to_double(), results_4[i]);
		assert_eq((a8).to_double(), results_8[i]);
		assert_eq((a16).to_double(), results_16[i]);
		assert_eq((a32).to_double(), results_32[i]);
		assert_eq((a64).to_double(), results_64[i]);
		std::cout << " ✓" << std::endl;
	}
}

void test_multiplication_base() {
	double vals_a[] = {1, 1, 1, -1, -1, -1, 0, 0, 0};
	double vals_b[] = {1, -1, 0, 1, -1, 0, 1, -1, 0};

	std::cout << "=== Test basic multiplication ===" << std::endl;
	for (auto a : vals_a) {
		auto a4(Posit4{a});
		auto a8(Posit8{a});
		auto a16(Posit16{a});
		auto a32(Posit32{a});
		auto a64(Posit64{a});

		for (auto b : vals_b) {
			std::cout << "\tTesting " << a << " * " << b << std::flush;
			double mul = a * b;
			auto b4(Posit4{b});
			auto b8(Posit8{b});
			auto b16(Posit16{b});
			auto b32(Posit32{b});
			auto b64(Posit64{b});
			assert_eq((a4 * b4).to_double(), mul);
			assert_eq((a8 * b8).to_double(), mul);
			assert_eq((a16 * b16).to_double(), mul);
			assert_eq((a32 * b32).to_double(), mul);
			assert_eq((a64 * b64).to_double(), mul);
			std::cout << " ✓" << std::endl;
		}
	}
}

void test_multiplication_easy() {
	double vals_a[] = {1.5, 1, 0.25, -0.5, -1, -1.25, 0, 0, 0};
	double vals_b[] = {1, -1.5, 0, 1, -1, 0, 1, -1, 0};

	std::cout << "=== Test basic multiplication ===" << std::endl;
	for (auto a : vals_a) {
		auto a8(Posit8{a});
		auto a16(Posit16{a});
		auto a32(Posit32{a});
		auto a64(Posit64{a});

		for (auto b : vals_b) {
			std::cout << "\tTesting " << a << " * " << b << std::flush;
			double mul = a * b;
			auto b8(Posit8{b});
			auto b16(Posit16{b});
			auto b32(Posit32{b});
			auto b64(Posit64{b});
			assert_eq((a8 * b8).to_double(), mul);
			assert_eq((a16 * b16).to_double(), mul);
			assert_eq((a32 * b32).to_double(), mul);
			assert_eq((a64 * b64).to_double(), mul);
			std::cout << " ✓" << std::endl;
		}
	}
}

void test_multiplication_edge() {
	std::cout << "=== Test multiplication cases ===" << std::endl;
	std::cout << "\tTesting -1.25 * 10" << std::flush;
	assert_eq((Posit8(-1.25) * Posit8(10.0)).to_double(), -12.0);
	assert_eq((Posit16(-1.25) * Posit16(10.0)).to_double(), -12.5);
	std::cout << " ✓" << std::endl;

	std::cout << "\tTesting 1 * 0.3" << std::flush;
	assert_eq((Posit8(1.0) * Posit8(0.3)).to_double(), Posit8(static_cast<uint8_t>(0b00110010)).to_double());
	assert_eq((Posit16(1.0) * Posit16(0.3)).to_double(), 0.300048828125);
	assert_eq((Posit32(1.0) * Posit32(0.3)).to_double(), 0.30000000074505806);
	assert_eq((Posit64(1.0) * Posit64(0.3)).to_double(), 0.3);
	std::cout << " ✓" << std::endl;
}

int main() {
	// double vals[] = {1337};
	test_conversion();
	test_multiplication_base();
	test_multiplication_easy();
	test_multiplication_edge();
	std::cout << std::endl;
	if (global_fails) {
		std::cout << "Failed " << global_fails << " test cases." << std::endl;
		return EXIT_FAILURE;
	} else {
		std::cout << "OK" << std::endl;
	}
#if 0
	double vals[] = {0, 0.0 / 0.0, 1337, 0.1337, -1, -1.5, -0.5, -0.75, -0.1337, 0.123456789, -0.123456789};
	for (auto val : vals) {
		Posit4 a{val};
		Posit8 b{val};
		Posit16 c{val};
		Posit32 d{val};
		Posit64 e{val};
		// a.print_info();
		printf("%.16lf is %.16lf\n", val, a.to_double());
		printf("%.16lf is %.16lf\n", val, b.to_double());
		// c.print_info();
		printf("%.16lf is %.16lf\n", val, c.to_double());
		printf("%.16lf is %.16lf\n", val, d.to_double());
		// e.print_info();
		printf("%.16lf is %.16lf\n", val, e.to_double());
		printf("============================\n");
	}
#endif
}
