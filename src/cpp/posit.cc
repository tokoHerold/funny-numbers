#include "posit.h"

#include <bit>
#include <cassert>
#include <cmath>
#include <compare>
#include <cstdint>

template <int N>
Posit<N>::Posit(double d) : bits(0) {
	static_assert(N <= 64);
	constexpr int n_ = N - 1;
	const double MAX_POS = std::pow(2.0, 4.0 * N - 8);
	const double MIN_POS = std::pow(2.0, -4.0 * N + 8);
	/* === Special Cases === */
	if (d == 0) return;  // Zero
	if (d == NAN || d == INFINITY || d == -INFINITY) {
		bits |= 1ull << n_;  // Not a Real (NaR)
		return;
	}
	if (std::abs(d) >= MAX_POS) {
		bits |= (1ull << n_) - 1;  // Max. posit
		return;
	}
	if (std::abs(d) <= MIN_POS) {
		bits |= 1ull;  // Min. posit
		return;
	}
	/* === General Case === */
	auto float_bits = std::bit_cast<uint64_t>(d);
	int exponent = static_cast<int>((float_bits >> 52) & 0x7FFll);  // 11 Bits
	uint64_t mantissa = float_bits & 0xF'FFFF'FFFF'FFFFll;          // 52 Bits
	int total_exponent;
	uint64_t fraction;
	int frac_len{52};
	/* --- Extract fraction (f) */
	if (exponent == 0) {                                // Subnormal: (-1)^s * 0.mant * 2^-1022
		int msb_pos = 64 - std::countl_zero(mantissa);  // find pos of MSB in mantissa
		total_exponent = msb_pos - 1074;                // -1022 - 52 + shift
		fraction = mantissa & ((1ll << msb_pos) - 1);   // Convert subnormal to normal
		frac_len = msb_pos - 1;
	} else {  // Normal: (-1^s) * 1.man * 2^(exp - 1023)
		fraction = mantissa;
		total_exponent = exponent - 1023;
	}
	constexpr int STORAGE_BITS = sizeof(storage_t) * 8;
	auto shift = std::max(frac_len - STORAGE_BITS, 0);
	auto clipped_fraction = static_cast<storage_t>(fraction >> shift);
	set_and_round(*this, (float_bits >> 63), total_exponent >> 2, total_exponent & 3, clipped_fraction);
}

template <int N>
constexpr std::strong_ordering Posit<N>::operator<=>(const Posit& other) const {
	if constexpr (SHIFT_ALIGN == 0) {  // Optimization for standard sizes (8, 16, 32, 64)
		return sbits <=> other.sbits;
	} else {  // For non-standard sizes, shift left to align sign bit, then compare.
		auto lhs = static_cast<signed_storage_t>(bits << SHIFT_ALIGN);
		auto rhs = static_cast<signed_storage_t>(other.bits << SHIFT_ALIGN);
		return lhs <=> rhs;
	}
}

template <int N>
std::tuple<int, int, typename Posit<N>::storage_t> Posit<N>::get_components() const {
	constexpr int BITS = std::numeric_limits<storage_t>::digits;
	auto abs = *this;
	if (get_sign_bit()) {
		abs = -abs;
	}
	int r, k, e{0};
	storage_t f{0};
	auto scratch = static_cast<storage_t>(abs.bits << SHIFT_ALIGN);  // Align posit bits with MSB
	scratch <<= 1;                                                   // Remove sign bit
	if (scratch & (static_cast<storage_t>(1) << (BITS - 1))) {       // Read direction
		k = std::countl_one(scratch);
		r = k - 1;
	} else {
		k = std::countl_zero(scratch);
		r = -k;
	}
	if (k + 1 >= BITS) return {r, e, f};
	scratch <<= (k + 1);  // Shift out regime
	e = static_cast<int>(scratch >> (BITS - 2));
	scratch <<= 2;                        // Shift out exponent
	int bits_consumed = 1 + (k + 1) + 2;  // Sign + Regime + Exp
	int frac_len = N - bits_consumed;
	if (frac_len > 0) {
		f = scratch >> (BITS - frac_len);
	}
	return {r, e, f};
}

template <int N>
double Posit<N>::to_double() {
	constexpr int BITS = std::numeric_limits<storage_t>::digits;
	if (bits == 0) {
		return +0;
	}
	if (*this == NAR) {
		return NAN;
	}
	auto [r, e, f] = get_components();
	auto tmp = static_cast<double>(1ll << (BITS - std::countl_zero(f)));
	double d = static_cast<double>(f) / tmp;
	d = (1.0 + d) * std::pow(2.0, 4.0 * r + e);
	return get_sign_bit() ? -d : d;
}

template <int N>
static void set_and_round(Posit<N>& posit, int s, int r, int e, typename Posit<N>::storage_t f) {
	// Automatically deduce the storage type U from the Posit definition
	using U = typename Posit<N>::storage_t;
	set_and_round_impl<N, U>(posit, s, r, e, f);
}

/**
 * @brief Primary logic implementation.
 * Expects N (Posit Width) to match U (Storage Width) exactly.
 * Used for N = 8, 16, 32, 64.
 */
template <int N, typename U>
static void set_and_round_impl(Posit<N>& posit, int s, int r, int e, U f) {
	// This implementation assumes the storage container fits the data exactly
	// to allow for native overflow behavior (simulating modulo arithmetic).
	static_assert(N == std::numeric_limits<U>::digits, "Primary impl requires N == BITS");

	assert(0 <= e && e < 4);
	assert(0 <= s && s < 2);

	constexpr int BITS = N;
	constexpr U ONE = static_cast<U>(1);
	constexpr U ALL_ONES = ~static_cast<U>(0);

	// Size of posit: | S | regime | exponent | fraction
	//       in bits: | 1 | k + 1  |     2    |    m
	int64_t k = (r >= 0) ? static_cast<int64_t>(r) + 1 : -static_cast<int64_t>(r);
	int m = (f == 0) ? 0 : BITS - std::countr_zero(f);

	U u = 0;  // u: N-Bit posit storage

	// --- Fast Path: Exact Fit ---
	if (k + m + 4 <= N) {                                 // (1 + k+1 + 2 + m <= N)
		U regime_bits = (r >= 0) ? ((ONE << k) - 1) << 1  // 111...10
		                         : ONE;                   // 000...01

		u |= regime_bits << (N - k - 2);
		u |= static_cast<U>(e) << (N - k - 4);
		if (m > 0) u |= f << (N - k - m - 4);
	}
	// --- General Path: Rounding ---
	else {
		U posit_v = 0;        // N+1 bit posit buffer (LSB is Guard, MSB is Regime Start)
		int pos = N - 1;      // Pointer to next unused space
		bool sticky = false;  // Determines whether N+1 Posit is above midpoint between two neighbouring N-Bit posits

		// |regime |exponent|fraction|
		//  ^
		// pos
		if (r >= 0) {  // Place Regime
			if (k >= N) {
				posit_v = ALL_ONES;
				pos = -1;                  // No space left
				if (k > N) sticky = true;  // Implicit 1s continue -> Above midpoint
			} else {
				U ones = (ONE << k) - 1;
				posit_v |= ones << (pos - k + 1);
				pos -= (k + 1);
			}
		} else {
			if (k >= N) {                  // All zeros
				pos = -1;                  // No space left
				if (k > N) sticky = true;  // Terminator 1 shifted out -> Above midpoint
			} else {
				pos -= k;
				posit_v |= (ONE << pos);
				pos -= 1;
			}
		}
		// |regime|exponent|fraction
		//         ^
		//        pos
		if (pos >= 0) {      // Place Exponent
			if (pos >= 1) {  // Enough space to place entire exponent
				posit_v |= static_cast<U>(e) << (pos - 1);
				pos -= 2;
			} else {                                // pos == 0, can only place MSB of exponent
				posit_v |= static_cast<U>(e >> 1);  // Place at bit 0 (Guard position)
				if (e & 1) sticky = true;           // LSB of exponent lost
				pos -= 1;
			}
		} else {                        // Cannot place exponent
			if (e != 0) sticky = true;  // If exponent is not 0 -> above midpoint
		}
		// |regime|exponent|fraction
		//                 ^
		//                 pos
		if (pos >= 0 && m > 0) {  // Place Fraction (if it is not empty)
			if (m <= pos + 1) {   // fraction fits exactly into the remaining space
				posit_v |= f << (pos - m + 1);
			} else {  // Truncate fraction
				int shift = BITS - (pos + 1);
				posit_v |= f >> shift;
				if ((f & ((ONE << shift) - 1)) != 0) sticky = true;
			}
		} else if (m > 0) {
			sticky = true;
		}
		// Rounding
		bool guard = (posit_v & 1);
		bool lsb = (posit_v & 2);
		if (guard && (sticky || lsb) && posit_v != ALL_ONES) posit_v++;  // Round up
		u = posit_v >> 1;                                                // Make space for sign
	}
	if (s == 1) u = -u;  // Apply Sign
	posit.bits = u;
}

/**
 * @brief Specialization for Posit4 (N=4, U=uint8_t).
 * Handles N < BITS mismatch requiring masking.
 */
template <>
void set_and_round_impl<4, uint8_t>(Posit<4>& posit, int s, int r, int e, uint8_t f) {
	constexpr int N = 4;
	using U = uint8_t;
	constexpr U MASK = 0x0F;  // Low 4 bits

	assert(0 <= e && e < 4);
	assert(0 <= s && s < 2);

	int64_t k = (r >= 0) ? static_cast<int64_t>(r) + 1 : -static_cast<int64_t>(r);
	int m = (f == 0) ? 0 : 8 - std::countl_zero(f);

	U u = 0;

	// General Path (Fast path mathematically impossible for N=4 due to k+m+4 > 4)
	U posit_v = 0;
	int pos = N - 1;
	bool sticky = false;

	// |regime |exponent|fraction|
	//  ^
	// pos
	if (r >= 0) {  // Place Regime
		if (k >= N) {
			posit_v = MASK;
			pos = -1;
			if (k > N) sticky = true;
		} else {
			U ones = (1 << k) - 1;
			posit_v |= ones << (pos - k + 1);
			pos -= (k + 1);
		}
	} else {
		if (k >= N) {
			pos = -1;
			if (k > N) sticky = true;
		} else {
			pos -= k;
			posit_v |= (1 << pos);
			pos -= 1;
		}
	}

	// |regime|exponent|fraction
	//         ^
	//        pos
	if (pos >= 0) {  // Place Exponent
		if (pos >= 1) {
			posit_v |= static_cast<U>(e) << (pos - 1);
			pos -= 2;
		} else {
			posit_v |= static_cast<U>(e >> 1) << pos;
			if (e & 1) sticky = true;
			pos -= 1;
		}
	} else {
		if (e != 0) sticky = true;
	}

	// |regime|exponent|fraction
	//                 ^
	//                 pos
	if (pos >= 0 && m > 0) {  // Place Fraction
		if (m <= pos + 1) {
			posit_v |= f << (pos - m + 1);
		} else {
			int shift = m - (pos + 1);
			posit_v |= f >> shift;
			if ((f & ((1 << shift) - 1)) != 0) sticky = true;
		}
	} else if (m > 0) {
		sticky = true;
	}

	// Rounding
	bool guard = (posit_v & 1);
	bool lsb = (posit_v & 2);

	if (guard && (sticky || lsb)) {
		if (posit_v != MASK) {
			posit_v++;
		}
	}

	u = posit_v >> 1;

	// Apply Sign
	if (s == 1) {
		u = ((~u) + 1) & MASK;
	} else {
		u &= MASK;
	}

	posit.bits = u;
}

// Explicitly instantiate for N=64 so the linker can find it
template struct Posit<64>;
template struct Posit<32>;
template struct Posit<16>;
template struct Posit<8>;
template struct Posit<4>;
