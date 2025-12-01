#pragma once
#include <cuda_runtime.h>

#include <cstdint>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// --- Storage Type Mapping ---
template <int N>
struct StorageType;
template <>
struct StorageType<64> {
	using type = uint64_t;
	using stype = int64_t;
};
template <>
struct StorageType<32> {
	using type = uint32_t;
	using stype = int32_t;
};
template <>
struct StorageType<16> {
	using type = uint16_t;
	using stype = int16_t;
};
template <>
struct StorageType<8> {
	using type = uint8_t;
	using stype = int8_t;
};
template <>
struct StorageType<4> {
	using type = uint8_t;
	using stype = int8_t;
};

// --- Math Helpers ---

// Count Leading Zeros (width agnostic)
template <typename T>
__device__ int gpu_clz(T val);
template <>
__device__ inline int gpu_clz<uint64_t>(uint64_t val) {
	return (val == 0) ? 64 : __clzll(val);
}
template <>
__device__ inline int gpu_clz<uint32_t>(uint32_t val) {
	return (val == 0) ? 32 : __clz(val);
}
template <>
__device__ inline int gpu_clz<uint16_t>(uint16_t val) {
	return (val == 0) ? 16 : (__clz((uint32_t)val) - 16);
}
template <>
__device__ inline int gpu_clz<uint8_t>(uint8_t val) {
	return (val == 0) ? 8 : (__clz((uint32_t)val) - 24);
}

// Safe Shift Right (Handles shift >= width)
template <typename T>
__device__ inline T safe_shr(T val, int shift) {
	constexpr int BITS = sizeof(T) * 8;
	if (shift >= BITS) return 0;
	if (shift < 0) return 0;  // Should not happen in valid logic
	return val >> shift;
}

// Safe Shift Left
template <typename T>
__device__ inline T safe_shl(T val, int shift) {
	constexpr int BITS = sizeof(T) * 8;
	if (shift >= BITS) return 0;
	return val << shift;
}

// Increment (Modulo arithmetic based on N)
template <int N, typename T>
__device__ inline T increment(T val) {
	constexpr T MASK = (sizeof(T) * 8 == N) ? static_cast<T>(~0) : (static_cast<T>(1) << N) - 1;
	return (val + 1) & MASK;
}

// Twos Complement (Modulo arithmetic based on N)
template <int N, typename T>
__device__ inline T twos_complement(T val) {
	constexpr T MASK = (sizeof(T) * 8 == N) ? static_cast<T>(~0) : (static_cast<T>(1) << N) - 1;
	return ((~val) + 1) & MASK;
}

// --- Core Posit Logic (Replicating posit.cc) ---

template <int N, typename T>
__device__ void gpu_set_and_round(T& posit_bits, bool s, int r, int e, T f) {
	constexpr int BITS = sizeof(T) * 8;
	constexpr int SHIFT_ALIGN = BITS - N;
	constexpr T ONE = static_cast<T>(1) << (BITS - 1);
	constexpr T ALL_ONES = ~static_cast<T>(0);

	// Clamp r
	if (r < -BITS) r = -BITS;
	if (r > BITS - 1) r = BITS - 1;

	int64_t k = (r >= 0) ? static_cast<int64_t>(r) + 1 : -static_cast<int64_t>(r);

	T regime_bits;
	if (r >= 0) {
		// '1' * k + '0'. If k >= BITS, result is all ones.
		regime_bits = (k >= BITS) ? ALL_ONES : (ALL_ONES << (BITS - k));
	} else {
		// '0' * k + '1'.
		regime_bits = (k >= BITS) ? 0 : (ONE >> k);
	}

	T posit_v = 0;
	posit_v |= regime_bits;

	// Exponent: shift = BITS - k - 3 (standard logic from posit.cc)
	// We must handle negative shifts safely (though strictly they shift out)
	int exp_shift = BITS - k - 3;
	if (exp_shift >= 0 && exp_shift < BITS) {
		posit_v |= static_cast<T>(e) << exp_shift;
	}

	// Fraction: shift = k + 3
	posit_v |= safe_shr(f, (int)(k + 3));

	// Final shift to align N bits
	T posit_u = safe_shr(posit_v, 1 + SHIFT_ALIGN);

	// Rounding
	constexpr T GUARD_SELECTOR = ONE >> (N - 1);
	bool lsb = (posit_u & 1);
	bool guard = (posit_v & GUARD_SELECTOR);

	if (guard) {
		bool round_up = lsb;

		// Check lost fraction precision
		int mask_shift = SHIFT_ALIGN + k + 3;
		if (mask_shift < BITS) {
			T mask = (static_cast<T>(1) << mask_shift) - 1;
			if ((mask & f) != 0) round_up = true;
		} else {
			// If the shift amount covers the whole type, any non-zero fraction implies loss
			if (f != 0) round_up = true;
		}

		// Check lost exponent precision
		int e_check_shift = BITS - k - 3;
		if (e_check_shift < 0) e_check_shift = 0;  // Cap at 0 to check all bits if needed
		// The original logic checks bits *below* the ones we successfully packed
		// Actually, logic is: (e >> max(..., 0)) != 0
		if (e_check_shift < BITS && (e >> e_check_shift) != 0) {
			round_up = true;
		}

		if (round_up) {
			posit_u = increment<N>(posit_u);
		}
	}

	if (s) posit_u = twos_complement<N>(posit_u);
	posit_bits = posit_u;
}

template <int N, typename T>
__device__ void gpu_get_components(T bits, int& r, int& e, T& f, bool& s) {
	constexpr int BITS = sizeof(T) * 8;
	constexpr int SHIFT_ALIGN = BITS - N;
	constexpr T MSB = static_cast<T>(1) << (BITS - 1);

	T abs_bits = bits;
	s = (bits & (static_cast<T>(1) << (N - 1))) != 0;
	if (s) abs_bits = twos_complement<N>(bits);

	// Align
	T scratch = safe_shl(abs_bits, SHIFT_ALIGN);
	scratch <<= 1;  // Remove sign bit

	int k;
	if (scratch & MSB) {
		// Leading ones
		k = gpu_clz(static_cast<T>(~scratch));
		r = k - 1;
	} else {
		// Leading zeros
		k = gpu_clz(scratch);
		r = -k;
	}

	if (k + 1 >= BITS) {
		e = 0;
		f = 0;
		return;
	}

	scratch <<= (k + 1);
	e = static_cast<int>(scratch >> (BITS - 2));
	scratch <<= 2;
	f = scratch;
}

// --- Arithmetic Operations ---

template <int N, typename T>
__device__ T gpu_posit_mul(T a_bits, T b_bits) {
	using ST = typename StorageType<N>::stype;
	constexpr int BITS = sizeof(T) * 8;
	constexpr T MSB_MASK = static_cast<T>(1) << (BITS - 1);
	constexpr T NAR = static_cast<T>(1) << (N - 1);

	if (a_bits == 0 || b_bits == 0) return 0;
	if (a_bits == NAR || b_bits == NAR) return NAR;

	int r1, e1, r2, e2;
	T f1, f2;
	bool s1, s2;

	gpu_get_components<N, T>(a_bits, r1, e1, f1, s1);
	gpu_get_components<N, T>(b_bits, r2, e2, f2, s2);

	int s_res = s1 ^ s2;
	ST exp_res = (static_cast<ST>(4) * r1 + e1) + (static_cast<ST>(4) * r2 + e2);

	// Mantissa Setup
	constexpr T MSB_1 = static_cast<T>(1) << (BITS - 1);
	T m1 = MSB_1 | (f1 >> 1);
	T m2 = MSB_1 | (f2 >> 1);

	// --- Manual Multiplication to match CPU exactly ---
	constexpr int H = BITS / 2;
	constexpr T LOW_MASK = (static_cast<T>(1) << H) - 1;

	T m1_lo = m1 & LOW_MASK;
	T m1_up = m1 >> H;
	T m2_lo = m2 & LOW_MASK;
	T m2_up = m2 >> H;

	T p_mid1 = m1_lo * m2_up;
	T p_mid2 = m1_up * m2_lo;
	T p_upper = m1_up * m2_up;

	T p_mid = p_mid1 + p_mid2;
	T mid_carry = (p_mid < p_mid1) ? (static_cast<T>(1) << H) : 0;

	// CPU logic: p_upper + (p_mid >> H) + mid_carry
	T m_prod = p_upper + (p_mid >> H) + mid_carry;

	// Normalize
	if ((m_prod & MSB_MASK) == 0) {
		m_prod <<= 1;
	} else {
		exp_res++;
	}

	T f_final = m_prod & (~MSB_MASK);

	int r_final = static_cast<int>(exp_res >> 2);
	int e_final = static_cast<int>(exp_res & 3);
	f_final <<= 1;

	T result;
	gpu_set_and_round<N, T>(result, s_res, r_final, e_final, f_final);
	return result;
}

template <int N, typename T>
__device__ T gpu_posit_add(T a_bits, T b_bits) {
	using ST = typename StorageType<N>::stype;
	constexpr int BITS = sizeof(T) * 8;
	constexpr T MSB_MASK = static_cast<T>(1) << (BITS - 1);
	constexpr T NAR = static_cast<T>(1) << (N - 1);

	if (a_bits == 0) return b_bits;
	if (b_bits == 0) return a_bits;
	if (a_bits == NAR || b_bits == NAR) return NAR;

	int r1, e1, r2, e2;
	T f1, f2;
	bool s1, s2;
	gpu_get_components<N, T>(a_bits, r1, e1, f1, s1);
	gpu_get_components<N, T>(b_bits, r2, e2, f2, s2);

	ST exp1 = static_cast<ST>(4) * r1 + e1;
	ST exp2 = static_cast<ST>(4) * r2 + e2;

	T m1 = MSB_MASK | (f1 >> 1);
	T m2 = MSB_MASK | (f2 >> 1);

	if (exp1 < exp2) {  // Swap
		T tm = m1;
		m1 = m2;
		m2 = tm;
		bool ts = s1;
		s1 = s2;
		s2 = ts;
		ST te = exp1;
		exp1 = exp2;
		exp2 = te;
	}

	ST shift = exp1 - exp2;
	T sticky = 0;

	if (shift < BITS) {
		T mask = (static_cast<T>(1) << shift) - 1;
		if ((m2 & mask) != 0) sticky = 1;
		m2 = (m2 >> shift) | sticky;
	} else {
		sticky = (m2 != 0) ? 1 : 0;
		m2 = sticky;
	}

	T m_res = 0;
	bool res_sign = s1;
	bool overflow = false;

	if (s1 == s2) {
		m_res = m1 + m2;
		if (m_res < m1) overflow = true;
	} else {
		if (m1 >= m2) {
			m_res = m1 - m2;
		} else {
			m_res = m2 - m1;
			res_sign = !s1;
		}
	}

	if (m_res == 0) return 0;

	if (overflow) {
		T s_bit = m_res & 1;
		m_res = (m_res >> 1) | s_bit;
		m_res |= MSB_MASK;
		exp1++;
	} else {
		int lz = gpu_clz(m_res);
		if (lz > 0) {
			m_res <<= lz;
			exp1 -= lz;
		}
	}

	T f_final = m_res << 1;
	int r_final = static_cast<int>(exp1 >> 2);
	int e_final = static_cast<int>(exp1 & 3);

	T result;
	gpu_set_and_round<N, T>(result, res_sign, r_final, e_final, f_final);
	return result;
}
