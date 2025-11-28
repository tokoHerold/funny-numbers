#pragma once

#include <cstdint>
#include <vector>

#include "posit.h"

/**
 * @brief A wrapper class containing a std::vector of the desired posit type.
 *
 * The array subscript operator retrieves the Posit at the desired index.
 */
template <typename Posit>
struct PositArray {
	std::vector<Posit> payload;

	explicit PositArray(size_t size) : payload(size) {}

	Posit operator[](size_t index) const { return payload[index]; }

	Posit* data() { return payload.data(); }
};

/**
 * @brief A container storing two 4-bit Posits in a single byte.
 *
 * Standard C++ types must align to at least 1 byte. To achieve 4-bit density in
 * memory arrays, this helper manages packing/unpacking.
 */
struct PackedPosit4Pair {
	uint8_t raw;

	/**
	 * @brief Default constructor.
	 */
	constexpr PackedPosit4Pair() : raw(0) {}
	constexpr PackedPosit4Pair(uint8_t byte) : raw(byte) {}

	/**
	 * @brief Construct from two independent Posit4 values.
	 * @param lower The posit to store in the lower nibble.
	 * @param upper The posit to store in the upper nibble.
	 */
	constexpr PackedPosit4Pair(Posit4 lower, Posit4 upper) { raw = (lower.bits & Posit4::MASK) | (upper.bits << 4); }

	/**
	 * @brief Extract the lower 4-bit Posit.
	 */
	inline Posit4 get_lower() const { return Posit4(static_cast<uint8_t>(raw & 0x0F)); }

	/**
	 * @brief Extract the upper 4-bit Posit.
	 */
	inline Posit4 get_upper() const { return Posit4(static_cast<uint8_t>(raw >> 4)); }

	/**
	 * @brief Set the lower 4-bit Posit.
	 */
	inline void set_lower(Posit4 p) { raw = (raw & 0xF0) | (p.bits & Posit4::MASK); }

	/**
	 * @brief Set the upper 4-bit Posit.
	 */
	void set_upper(Posit4 p) { raw = (raw & 0x0F) | (p.bits << 4); }
};

static_assert(sizeof(PackedPosit4Pair) == 1, "PackedPosit4Pair is not 1 byte in size!");

// Define PositArray types:
template <>
struct PositArray<Posit4> {
	std::vector<PackedPosit4Pair> payload;

	explicit PositArray(size_t size) : payload((size + 1) / 2) {}

	Posit4 operator[](size_t index) const {
		if (index & 1) return payload[index / 2].get_upper();
		return payload[index / 2].get_lower();
	}

	PackedPosit4Pair* data() { return payload.data(); }
};

using Posit64Array = PositArray<Posit64>;
using Posit32Array = PositArray<Posit32>;
using Posit16Array = PositArray<Posit16>;
using Posit8Array = PositArray<Posit8>;
using Posit4Array = PositArray<Posit4>;
