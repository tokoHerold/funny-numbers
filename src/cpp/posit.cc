#include "posit.h"

#include <algorithm>
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
		// bits |= 1ull << n_;  // Not a Real (NaR)
		*this = nar();
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
	int shift = frac_len - STORAGE_BITS;
	storage_t clipped_fraction;
	if (shift >= 0) {
		clipped_fraction = fraction >> shift;
	} else {
		clipped_fraction = fraction << abs(shift);
	}
	set_and_round(*this, (float_bits >> 63), total_exponent >> 2, total_exponent & 3, clipped_fraction);
}

template <int N>
constexpr std::strong_ordering Posit<N>::operator<=>(const Posit& other) const {
	if constexpr (SHIFT_ALIGN == 0) {  // Optimization for standard sizes (8, 16, 32, 64)
		return sbits <=> other.sbits;
	} else {  // For non-standard sizes, shift left to align sign bit, then compare.
		auto lhs = static_cast<sstorage_t>(bits << SHIFT_ALIGN);
		auto rhs = static_cast<sstorage_t>(other.bits << SHIFT_ALIGN);
		return lhs <=> rhs;
	}
}

template <int N>
Posit<N> Posit<N>::operator+=(const Posit<N>& other) {
	constexpr int BITS = std::numeric_limits<storage_t>::digits;
	constexpr storage_t MSB_MASK = static_cast<storage_t>(1) << (BITS - 1);

	//  Special Cases
	if (bits == 0) {
		*this = other;
		return *this;
	}
	if (other.bits == 0) return *this;
	if (*this == nar() || other == nar()) {
		*this = nar();
		return *this;
	}
	auto [r1, e1, f1] = this->get_components();
	auto [r2, e2, f2] = other.get_components();

	bool s1 = this->get_sign_bit();
	bool s2 = other.get_sign_bit();

	// Unify exponent: Exp = 4*r + e
	sstorage_t exp1 = static_cast<sstorage_t>(4) * static_cast<sstorage_t>(r1) + static_cast<sstorage_t>(e1);
	sstorage_t exp2 = static_cast<sstorage_t>(4) * static_cast<sstorage_t>(r2) + static_cast<sstorage_t>(e2);

	// Mantissas: Add Implicit 1 - Shift right logic 1 to place hidden bit at MSB.
	storage_t m1 = MSB_MASK | (f1 >> 1);
	storage_t m2 = MSB_MASK | (f2 >> 1);

	// Align exponents
	sstorage_t shift;
	if (exp1 < exp2) {
		std::swap(m1, m2);
		std::swap(s1, s2);
		std::swap(exp1, exp2);
	}
	shift = exp1 - exp2;
	storage_t sticky =  // Round up if all bits are 1
	    shift < BITS && (m2 & ((static_cast<storage_t>(1) << shift) - 1)) ? 1 : 0;
	m2 = (m2 >> shift) | sticky;

	// Add/Sub
	storage_t m_res = 0;
	int res_sign = s1;
	bool overflow = false;

	if (s1 == s2) {  // Handle signs
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

	if (m_res == 0) {
		bits = static_cast<storage_t>(0);
		return *this;
	}

	// Normalize (on overflow)
	if (overflow) {
		storage_t sticky = (m_res & 1);
		m_res = (m_res >> 1) | sticky;
		m_res |= MSB_MASK;  // The carry bit becomes MSB
		exp1++;
	} else {  // In case of subtraction check if mantissa needs to be aligned again
		int lz = std::countl_zero(m_res);
		if (lz > 0) {
			m_res <<= lz;
			exp1 -= lz;
		}
	}

	// Encode
	sstorage_t r_final = exp1 >> 2;
	sstorage_t e_final = exp1 & 3;
	storage_t f_final = m_res << 1;  // Extract Fraction (Remove hidden bit MSB)

	set_and_round(*this, res_sign, static_cast<int>(r_final), static_cast<int>(e_final), f_final);
	return *this;
}

template <int N>
Posit<N> Posit<N>::operator+(const Posit<N>& other) const {
	Posit res = *this;
	res += other;
	return res;
}

template <int N>
Posit<N> Posit<N>::operator*=(const Posit<N>& other) {
	constexpr int BITS = std::numeric_limits<storage_t>::digits;
	constexpr storage_t MSB_MASK = static_cast<storage_t>(1) << (BITS - 1);

	if (bits == 0 || other.bits == 0) {
		bits = static_cast<sstorage_t>(0);
		return *this;
	}
	if (*this == nar() || other == nar()) {
		*this = nar();
		return *this;
	}

	auto [r1, e1, f1] = this->get_components();
	auto [r2, e2, f2] = other.get_components();

	int s_res = (this->get_sign_bit() ^ other.get_sign_bit());

	// Formula:
	// P = (1.f1 * 2^E1) * (1.f2 * 2^E2)
	// E = (4r1 + e1) + (4r2 + e2)
	sstorage_t exp_res = (static_cast<sstorage_t>(4) * r1 + e1) + (4 * r2 + e2);

	// Attach implicit 1 to mantissas
	constexpr storage_t MSB_1 = static_cast<storage_t>(1) << (BITS - 1);
	storage_t m1 = MSB_1 | (f1 >> 1);
	storage_t m2 = MSB_1 | (f2 >> 1);

	// Multiply Mantissas: m1 * m2
	// Perform (sizeof(storage_t)*2)-bit multiplication 'in-place'
	constexpr int H = BITS / 2;
	constexpr storage_t LOW_MASK = (static_cast<storage_t>(1) << H) - 1;

	// (m1_up * H + m1_low) * (m2_up * H + m2_low) =
	//      (m1_low * m2_low)     % Ignored, precision will be lost anyway
	//    + (m1_low * m2_up * H)  % p_mid1
	//    + (m2_low * m1_up * H)  % p_mid2
	//    + (m1_up * m2_up)       % p_upper
	storage_t m1_lo = m1 & LOW_MASK;
	storage_t m1_up = m1 >> H;
	storage_t m2_lo = m2 & LOW_MASK;
	storage_t m2_up = m2 >> H;

	storage_t p_mid1 = m1_lo * m2_up;
	storage_t p_mid2 = m1_up * m2_lo;
	storage_t p_upper = m1_up * m2_up;

	storage_t p_mid = p_mid1 + p_mid2;  // Check if this statement overflows
	storage_t mid_carry = (p_mid < p_mid1) ? (static_cast<storage_t>(1) << H) : 0;

	storage_t m_prod = p_upper + (p_mid >> H) + mid_carry;

	// Normalize (Post-Normalization) - Result of 1.x * 1.y is in [1, 4).
	if ((m_prod & MSB_MASK) == 0) {
		m_prod <<= 1;  // shift 'left' to normalize (<< because right-alignment)
	} else {
		exp_res++;
	}

	storage_t f_final = m_prod & (~MSB_MASK);

	int r_final = exp_res >> 2;
	int e_final = exp_res & 3;

	f_final <<= 1;  // Remove implicit 1
	set_and_round(*this, s_res, static_cast<int>(r_final), static_cast<int>(e_final), f_final);
	return *this;
}

template <int N>
Posit<N> Posit<N>::operator*(const Posit<N>& other) const {
	Posit res = *this;
	res *= other;
	return res;
}

// template<int N>
// constexpr

/**
 * @brief Decodes the posit into its components: regime, exponent, and fraction.
 *
 * This function extracts the internal components of the posit value.
 * It handles negative values by taking the absolute value (two's complement)
 * before decoding.
 *
 * @tparam N The bit width of the posit.
 * @return A tuple containing:
 * - int r: The regime value.
 * - int e: The exponent value.
 * - storage_t f: The fraction bits. The fraction is returned right-aligned
 * in the integer container. Leading zeros are significant and implicit.
 */
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
	scratch <<= 2;  // Shift out exponent
	f = scratch;
	return {r, e, f};
}

template <int N>
double Posit<N>::to_double() {
	if (bits == 0) {
		return +0;
	}
	if (*this == nar()) {
		return NAN;
	}
	auto [r, e, f] = get_components();
	auto tmp = static_cast<double>(1ll << (N - 2));  // Make sure enough space left to cast to double
	double d = static_cast<double>(f >> 2) / tmp;    // No precision is lost, as sign+regime takes at least 3 bits
	d = (1.0 + d) * std::pow(2.0, 4.0 * r + e);
	return get_sign_bit() ? -d : d;
}

template <int N>
static void set_and_round(Posit<N>& posit, int s, int r, int e, typename Posit<N>::storage_t f) {
	// Automatically deduce the storage type U from the Posit definition
	using U = typename Posit<N>::storage_t;
	set_and_round_impl<N, U>(posit, s, r, e, f);
}

template <int N>
constexpr typename Posit<N>::storage_t increment(typename Posit<N>::storage_t val) {
	return ++val & Posit<N>::MASK;
}

template <int N>
constexpr typename Posit<N>::storage_t twos_complement(typename Posit<N>::storage_t val) {
	return ((~val) + 1) & Posit<N>::MASK;
}

/**
 * @brief Primary logic implementation.
 * Expects N (Posit Width) to match U (Storage Width) exactly.
 * Used for N = 8, 16, 32, 64.
 *
 * @param posit: Reference to the posit value
 * @param s: Sign bit, either false for 0 or true for 1
 * @param r: regime exponent
 * @param e: exponent, must be between 0 and 3
 * @param f: right-aligned fraction bits, so that the MSB is at position N-1<br>
 *           Example (32 bit): 0.625 | 31 31 30 29 28 ... 0 |<br>
 *                                   |  1  0  1  0  0 ... 0 |
 */
template <int N, typename U>
static void set_and_round_impl(Posit<N>& posit, bool s, int r, int e, U f) {
	// This implementation assumes the storage container fits the data exactly
	// to allow for native overflow behavior (simulating modulo arithmetic).

	assert(0 <= e && e < 4);
	constexpr auto SHIFT_ALIGN = Posit<N>::SHIFT_ALIGN;
	constexpr int BITS = std::numeric_limits<U>::digits;
	constexpr U ONE = static_cast<U>(1) << (BITS - 1);  // Leading one
	constexpr U ALL_ONES = ~static_cast<U>(0);
	r = std::clamp(r, -BITS, BITS - 1);

	int64_t k = (r >= 0) ? static_cast<int64_t>(r) + 1 : -static_cast<int64_t>(r);
	U regime_bits = r >= 0                        // Encode r in unary:
	                    ? ALL_ONES << (BITS - k)  // '1' * k + '0' (right aligned)
	                    : ONE >> k;               // '0' * k + '1' (right aligned)

	U posit_u{0};  // N bit posit
	// Posit u: | S | regime | exponent | fraction
	//    bits: | 1 | k + 1  |     2    |    m

	U posit_v{0};  // N+1 bit posit buffer (LSB is Guard, MSB is Regime Start)
	// Posit v: | regime | exponent | fraction
	//    bits: | k + 1  |     2    |    m
	posit_v |= regime_bits;
	posit_v |= static_cast<U>(e) << (BITS - k - 3);
	posit_v |= f >> (k + 3);              // k+1 (regime) + 2 (exponent)
	posit_u = posit_v >> (1 + BITS - N);  // Make space for sign

	// Check if rounding up is required
	constexpr U GUARD_SELECTOR = ONE >> (N - 1);
	bool lsb = (posit_u & static_cast<U>(1)) > 0;
	bool guard = (posit_v & GUARD_SELECTOR) > 0;
	if (guard) {                                                              // Might be above midpoint
		if (lsb                                                               // If lsb != 0: Round to next even posit
		    || (((static_cast<U>(1) << (SHIFT_ALIGN + k + 3)) - 1) & f) != 0  // If precision in the fraction was lost
		    || (e >> std::max((BITS - k - 3), 0l)) != 0) {                    // If precision in the exponent was lost
			posit_u = increment<N>(posit_u);
		}
	}
	// int pos = N - 1;      // Pointer to next unused space
	// bool above_midpoint{false};
	if (s) posit_u = twos_complement<N>(posit_u);  // Apply Sign
	posit.bits = posit_u;
}

// Explicitly instantiate for N=64 so the linker can find it
template struct Posit<64>;
template struct Posit<32>;
template struct Posit<16>;
template struct Posit<8>;
template struct Posit<4>;
