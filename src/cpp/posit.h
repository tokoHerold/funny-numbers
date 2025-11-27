#pragma once

#include <bitset>
#include <compare>
#include <cstdint>
#include <iostream>
#include <type_traits>

#define NAR Posit<N>{static_cast<typename Posit<N>::storage_t>(1ll << (N-1))}

/**
 * @brief Automatically selects the smallest standard integer type that fits
 * NBITS.
 *
 * Ensures efficient memory usage by mapping bit widths to:
 * - 1-8 bits   -> uint8_t
 * - 9-16 bits  -> uint16_t
 * - 17-32 bits -> uint32_t
 * - 33-64 bits -> uint64_t
 */
template <int NBITS>
struct PositStorage {  // clang-format off
    using type =
        std::conditional_t<NBITS <= 8,  uint8_t,
        std::conditional_t<NBITS <= 16, uint16_t,
        std::conditional_t<NBITS <= 32, uint32_t,
        std::conditional_t<NBITS <= 64, uint64_t, 
        void>>>>;  // clang-format on

	static_assert(!std::is_same<type, void>::value, "Posit size too large (max 64 bits)");
};

/**
 * @brief A class representing a Posit number of N width.
 *
 * Implements parts of the 2. Posit Standard where ES (Exponent Size) is fixed
 to 2.
 * A posit is structured as follows:
 * Sign Bit | Regime Bits | Exponent Bits | Fraction Bits
        n-1..n-2 |  n-3..n-k   |  n-k-1..n-m   |   n-m-1..0

 * 1. Sign Bit (1 bit):
 *    0 for positive, 1 for negative.
 *    Negative numbers are represented using Two's Complement.
 *
 * 2. Regime Bits (Variable length):
 *    A run of identical bits terminated by an opposite bit.
 *    Determines the regime value (k), representing a coarse scale factor of
 16^k.
 *
 * 3. Exponent Bits (Up to 2 bits):
 *    The next 2 bits represent the unsigned exponent (e) over the base 2.
 *    This field is truncated if the regime consumes too many bits.
 *
 * 4. Fraction Bits (Remaining bits):
 *    The remaining bits form the fraction (f) of the significand 1.f.
 *
 * @tparam NBITS The total number of bits for the Posit (e.g., 64, 32, 16, 8,
 4).
 */
template <int N>
struct Posit {
	static constexpr int ES = 2;

	using storage_t = typename PositStorage<N>::type;
	using signed_storage_t = typename std::make_signed<storage_t>::type;

	union {
		storage_t bits;          // Underlying bits, unsigned
		signed_storage_t sbits;  // Underlying bits, signed
	};

	/**
	 * @brief Mask used for sub-byte sizes (e.g., 4-bit).
	 */
	static constexpr storage_t MASK =
	    (N == 8 * sizeof(storage_t)) ? static_cast<storage_t>(~0) : (static_cast<storage_t>(1) << N) - 1;
	// static constexpr Posit NAR{static_cast<storage_t>(1ll << (N - 1))};

	/**
	 * @brief The shift amount needed to align the Posit's sign bit with the storage MSB.
	 */
	static constexpr int SHIFT_ALIGN = (sizeof(storage_t) * 8) - N;

	/**
	 * @brief Default constructor. Initializes Posit to 0.
	 */
	constexpr Posit() : bits(0) {}

	/*
	 * @brief Copy constructor.
	 */
	constexpr Posit(const Posit& other) : bits(other.bits) {}

	/**
	 * @brief Construct from raw bits.
	 * @param raw_bits The integer representation of the bits.
	 */
	constexpr explicit Posit(storage_t raw_bits) : bits(raw_bits & MASK) {}

	/**
	 * @brief Construct from double.
	 *
	 * Converts an IEEE 754 floating point value to the nearest Posit representation.
	 * @param d The value to convert.
	 */
	explicit Posit(double d);

	// --- Operator Overloading ---
	constexpr Posit& operator=(const Posit& other) {
		bits = other.bits;
		return *this;
	}
	constexpr bool operator==(const Posit& other) const { return bits == other.bits; }
	constexpr std::strong_ordering operator<=>(const Posit& other) const;
	constexpr Posit operator-() const {
		if constexpr (sizeof(storage_t) == N) {
			return Posit<N>{static_cast<storage_t>((~bits) + 1)};
		}
		return Posit<N>{static_cast<storage_t>(((~bits) + 1) & MASK)};
	}
	Posit operator+(const Posit& other) const;
	Posit operator*(const Posit& other) const;

	// --- Functions ---
	constexpr bool get_sign_bit() const { return bits & (static_cast<storage_t>(1) << (N - 1)); }
	double to_double();
	std::tuple<int, int, storage_t> get_components() const;

	/**
	 * @brief Prints debug information to standard output.
	 */
	void print_info() const {
		std::cout << "Posit<" << N << ">: "
		          << "Raw: " << std::bitset<N>(bits)  // Use std::bitset to print N bits
		          << " (Signed: " << (int64_t)sbits << ")"
		          << " (Storage: " << sizeof(storage_t) * 8 << " bits)" << std::endl;
	}
};

using Posit64 = Posit<64>;
static_assert(sizeof(Posit64) == 8, "Size of Posit64 is not 8");
using Posit32 = Posit<32>;
static_assert(sizeof(Posit32) == 4, "Size of Posit32 is not 4");
using Posit16 = Posit<16>;
static_assert(sizeof(Posit16) == 2, "Size of Posit16 is not 2");
using Posit8 = Posit<8>;
static_assert(sizeof(Posit8) == 1, "Size of Posit8 is not 1");
using Posit4 = Posit<4>;
static_assert(sizeof(Posit4) == 1, "Size of Posit4 is not 1");

// // ==========================================
// // 4. Packed Storage Utilities
// // ==========================================
//
// /**
//  * @brief A compact container storing two 4-bit Posits in a single byte.
//  *
//  * Standard C++ types must align to at least 1 byte. To achieve true 4-bit density
//  * in memory arrays (e.g., for ML weights), this helper manages packing/unpacking.
//  */
// struct PackedPosit4Pair {
// 	/**
// 	 * @brief The raw byte containing two 4-bit posits.
// 	 * - Bits [0..3]: The "Lower" posit.
// 	 * - Bits [4..7]: The "Upper" posit.
// 	 */
// 	uint8_t raw_byte;
//
// 	/**
// 	 * @brief Default constructor.
// 	 */
// 	constexpr PackedPosit4Pair() : raw_byte(0) {}
//
// 	/**
// 	 * @brief Construct from two independent Posit4 values.
// 	 * @param lower The posit to store in the lower nibble.
// 	 * @param upper The posit to store in the upper nibble.
// 	 */
// 	constexpr PackedPosit4Pair(Posit4 lower, Posit4 upper) {
// 		raw_byte = (lower.bits & 0x0F) | ((upper.bits & 0x0F) << 4);
// 	}
//
// 	/**
// 	 * @brief Extract the lower 4-bit Posit.
// 	 */
// 	Posit4 get_lower() const { return Posit4(static_cast<uint8_t>(raw_byte & 0x0F)); }
//
// 	/**
// 	 * @brief Extract the upper 4-bit Posit.
// 	 */
// 	Posit4 get_upper() const { return Posit4(static_cast<uint8_t>((raw_byte >> 4) & 0x0F)); }
//
// 	/**
// 	 * @brief Update the lower 4-bit Posit.
// 	 */
// 	void set_lower(Posit4 p) { raw_byte = (raw_byte & 0xF0) | (p.bits & 0x0F); }
//
// 	/**
// 	 * @brief Update the upper 4-bit Posit.
// 	 */
// 	void set_upper(Posit4 p) { raw_byte = (raw_byte & 0x0F) | ((p.bits & 0x0F) << 4); }
// };
