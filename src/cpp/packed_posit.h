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

	Posit& operator[](size_t index) { return payload[index]; }

	const Posit& operator[](size_t index) const { return payload[index]; }

	Posit* data() { return payload.data(); }
};

/**
 * @brief A container storing two 4-bit Posits in a single byte.
 *
 * Standard C++ types must align to at least 1 byte. To achieve 4-bit density in
 * memory arrays, this helper manages packing/unpacking.
 */
union PackedPosit4Pair {
	uint8_t raw;
	struct {
		uint8_t lower : 4;
		uint8_t upper : 4;
	} nibbles;

	constexpr PackedPosit4Pair() : raw(0) {}
	constexpr PackedPosit4Pair(uint8_t byte) : raw(byte) {}

	/**
	 * @brief Construct from two independent Posit4 values.
	 */
	constexpr PackedPosit4Pair(Posit4 lower, Posit4 upper) { raw = (lower.bits & Posit4::MASK) | (upper.bits << 4); }

	/**
	 * @brief Extract the lower 4-bit Posit.
	 */
	inline Posit4 get_lower() const { return Posit4(nibbles.lower); }

	/**
	 * @brief Extract the upper 4-bit Posit.
	 */
	inline Posit4 get_upper() const { return Posit4(nibbles.upper); }

	/**
	 * @brief Set the lower 4-bit Posit.
	 */
	inline void set_lower(Posit4 p) { nibbles.lower = p.bits; }

	/**
	 * @brief Set the upper 4-bit Posit.
	 */
	inline void set_upper(Posit4 p) { nibbles.upper = p.bits; }
};

static_assert(sizeof(PackedPosit4Pair) == 1, "PackedPosit4Pair is not 1 byte in size!");

// Define PositArray types:
template <>
struct PositArray<Posit4> {
	std::vector<PackedPosit4Pair> payload;

	explicit PositArray(size_t size) : payload((size + 1) / 2) {}

	// --- Proxy Class for Assignment ---
	class Posit4Reference {
		PackedPosit4Pair& pair;
		bool is_upper;

	   public:
		Posit4Reference(PackedPosit4Pair& p, bool upper) : pair(p), is_upper(upper) {}

		// Assignment from Posit4 (The setter)
		Posit4Reference& operator=(Posit4 val) {
			if (is_upper)
				pair.set_upper(val);
			else
				pair.set_lower(val);
			return *this;
		}

		// Assignment from another reference
		Posit4Reference& operator=(const Posit4Reference& other) { return *this = static_cast<Posit4>(other); }
		Posit4Reference& operator+=(Posit4 val) {
			Posit4 current = *this;  // Convert to Posit4
			current += val;          // Perform op
			return *this = current;  // Assign back
		}

		Posit4Reference& operator*=(Posit4 val) {
			Posit4 current = *this;
			current *= val;
			return *this = current;
		}

		double to_double() const {
			Posit4 current = *this;
		   return current.to_double();	
		}

		// Implicit conversion to Posit4 (The getter)
		operator Posit4() const { return is_upper ? pair.get_upper() : pair.get_lower(); }
	};

	// --- Accessors ---

	// Mutable access returns the Proxy
	Posit4Reference operator[](size_t index) { return Posit4Reference(payload[index / 2], index & 1); }

	// Const access returns the value directly (read-only)
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
