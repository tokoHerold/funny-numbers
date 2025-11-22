import math
import struct
from typing import override


class Posit:
    """
    Implementation of a Posit number.

    Implements parts of the [Posit standard](https://posithub.org/about)
    """

    def __init__(self, value: float = 0.0, n: int = 32):
        """
        Initialize a Posit number with the given value, precision, and exponent size.

        A posit is structured as follows:
        Sign Bit | Regime Bits | Exponent Bits | Fraction Bits
        n-1..n-2 |  n-3..n-k   |  n-k-1..n-m   |   n-m-1..0

        The sign bit is 0 for positive values and 1 for negative values.
        If it is 1, the remaining bits are interpreted in negative two's complement.
        The regime bits consist of k-1 identical bits, followed by one that is inverted. k can be in the range [2, n-1].
        The exponent consists of max(min(n - k -1, 2), 0) bits.
        The fraction is the remainder of available bits.
        # TODO: write this better

        Args:
            value (float): The value of the Posit number.
            n (int): The precision of the Posit number, i.e. the number of total bits used.
        """
        if n < 2:
            raise ValueError("Posit precision n must be at least 2")

        self._bits: str = "0" * n
        """Bit sequence of the posit number"""

        self.BITWIDTH: int = n
        "Number of bits (n) used to represent the number."
        self.P_INT_MAX: int = int(math.ceil(2.0 ** math.floor(4 * (n - 3) / 5)))
        """The largest consecutive integer-valued posit value."""
        self.MIN_POS: float = 2.0 ** (-4 * n + 8)
        """The smallest positive posit value."""
        self.MAX_POS: float = 2.0 ** (4 * n - 8)
        """The largest positive posit value."""
        self._USEED: float = 16.0  # sepcified by posit standard v2

        self._float_to_posit(value)


    def _set_bits(self, bits: str):
        if len(bits) != self.BITWIDTH:
            raise ValueError(f"Expected {self.BITWIDTH} bits, got {len(bits)}")
        self._bits = bits


    def is_zero(self) -> bool:
        """Tests if the value of this posit is equal to 0.0"""
        return self._bits == '0' * self.BITWIDTH


    def is_nar(self) -> bool:
        """Tests if the value of this posit is not-a-real (NaR)"""
        return self._bits == '1' + '0' * (self.BITWIDTH - 1)


    def _float_to_posit(self, value: float):
        """Converts a 64-bit float into a n-bit posit"""
        n: int  = self.BITWIDTH

        # === Special Cases === #
        # Special case #1: zero
        if value == 0:
            self._set_bits("0" * n)
            return
        # Special case #2: nar
        if math.isnan(value) or math.isinf(value):
            self._set_bits("1" + "0" * (n - 1))
            return
        # Special case #3: float >= MAX_POS
        if value >= self.MAX_POS:
            self._set_bits('0' + '1' * (n - 1))
            return
        if abs(value) <= self.MIN_POS:
            self._set_bits('0' * (n - 1) + '1')
            return

        # === General Case === #
        # Pack to IEEE 754 double (64-bit)
        # [Sign (1)] [Exponent (11)] [Mantissa (52)]
        packed = struct.pack('>d', value)  # '>d' means Big-Endian Double
        float_bits = int.from_bytes(packed, 'big')
        ieee754_sign = (float_bits) >> 63 & 1  # Sign bit

        ieee754_exp = (float_bits >> 52) & 0x7FF  # (11 Bits)
        ieee754_mantissa = float_bits & 0xF_FFFF_FFFF_FFFF  # (52 Bits)

        # --- Extract Fraction (f) ---#
        if ieee754_exp == 0:  # - Subnormal: (-1)^s * 0.mant * 2^-1022
            # Normalize 0.mant to 1.f -> shift left until MSB is one
            bit_len = ieee754_mantissa.bit_length() - 1  # Leading 1 search
            total_exponent = bit_len - 1074  # 52 bits of fraction + 1022 exp

            f_int = ieee754_mantissa & ((1 << bit_len) - 1) # Shift mantissa left until MSB is 1
            f = format(f_int, f'0{bit_len}b') if bit_len > 0 else ""
        else:  # ---------------- Normal: (-1)^s * 1.mant * 2^(exp - 1023)
            total_exponent = ieee754_exp - 1023
            f = format(ieee754_mantissa, '052b')

        # --- Decompose total exponent into Regime (r) and Exponent (e) --- #
        # total_exponent = 4r + e
        r = total_exponent >> 2 # div 4
        e = total_exponent & 3  # mod 4

        # --- Encode parts into posit --- #
        self._encode_compnents(ieee754_sign, r, e, f)


    def _encode_compnents(self, s: int, r: int, e: int, f: str):
        """
        Encodes a Posit from its decoded components.

        Args:
            s (int): Sign (0 for positive, 1 for negative).
            r (int): Regime value (numerical exponent of useed).
            e (int): Exponent value (0-3).
            f (str): Fraction bits.
        """
        assert (0 <= s < 2) and (0 <= e < 4)
        n = self.BITWIDTH
        bit_str = '0'

        # === (n/n+x) bit Posit construction === #
        if r >= 0:
            k = r + 1
            bit_str += '1' * k + '0'
        else:
            k = -r
            bit_str += '0' * k + '1'
        bit_str += format(e, '02b')  # Binary string of length 2
        bit_str += f
        if len(bit_str) <= n:  # Value can be expressed as n-bit posit
            bit_str = bit_str.ljust(n, '0')
        else:  # === Posit rounding === #
            # Posit rounding works as follows:
            # let v be the n+1 bit posit currently stored in `posit_bits`
            # let u, w be the n bit posits, whose value surrounds v.
            u = bit_str[:n]  # next smaller n-bit posit
            # Because u, v and w have the same r, e incrementing u yields w.
            # As v has n+1 bits, LSB decides whether u or w are nearer.
            guard_bit = bit_str[n]  # n+1'th bit
            lsb = u[-1]
            above_midpoint = '1' in bit_str[n+1:]
            if guard_bit == '1' and (above_midpoint or lsb == '1'):
                # Above midpoint or round-to-even (Posit spec)
                bit_str = self._increment_bits(u)  #round up (v)
            else:
                bit_str = u  # Round down

        # === Apply sign bit === #
        if s == 0:
            self._set_bits(bit_str)
        else:
            self._set_bits(self._twos_complement(bit_str))


    def _get_components(self) -> tuple[int, int, int, str]:
        """
        Decodes the posit components:
            s (int): Sign (0 for positive, 1 for negative).
            r (int): Regime value (numerical exponent of useed).
            e (int): Exponent value (0-3).
            f (str): Fraction bits.

            Returns:
                tuple[int, int, int, str] of the form (s, r, e, f).
        """
        n = self.BITWIDTH
        bits = self._bits
        # === Read sign bit (s) === #
        s = 0 if self._bits[0] == '0' else 1
        #  === Parse regime (r) === #
        k = bits[1:].find('0' if bits[1] == '1' else '1')  # find run length
        if k == -1: # Regime is whole posit
            k = n-1
        r = -k if bits[1] == '0' else k - 1
        # === Read exponent (e) === #
        remaining = bits[2 + k:] # Stored in next 2 bits (spec)
        e = int(remaining[:2].ljust(2, '0'), 2)  # consider truncated exponent
        # === Read fraction (f) === #
        f = remaining[2:]
        return s, r, e, f


    def to_float(self) -> float:
        """Converts the value of this posit into a IEEE 754 float"""
        if self.is_zero():  # Special case 1
            return +0.0
        if self.is_nar():  # Special case 2
            return float('nan')
        s, r, e, f_str = self._get_components()
        f = 0.0  # Transform fixed point string into float
        if len(f_str) > 0:
            f = int(f_str.ljust(1, '0'), 2) / 2.0**len(f_str)
        return ((1 - 3 * s) + f) * 2.0**((1 - 2 * s) * (4 * r + e + s)) # See spec


    def __mul__(self, other: 'Posit') -> 'Posit':
        """
        Multiplies this Posit by another Posit.
        """
        if not isinstance(other, Posit):
            raise ValueError(f"__mul__: Expected Posit, got {type(other)}")

        # --- Handle Special Cases --- #
        if self.is_nar() or other.is_nar():
            return Posit(self.nbits, float('nan'))
        if self.is_zero() or other.is_zero():
            return Posit(self.nbits, 0.0)

        # --- Calculate components --- #
        s_self, r_self, e_self, f_self = self._get_components()
        s_other, r_other, e_other, f_other = other._get_components()
        # TODO



    @override
    def __str__(self) -> str:
        return str(self.to_float()) + 'p'


    @override
    def __repr__(self) -> str:
        return f'Posit{self.BITWIDTH} ({self._bits})\n{self.to_float()}'


    @staticmethod
    def _twos_complement(bits: str) -> str:
        """Calculates the two's complement of a bit string of arbitrary length."""
        if "1" not in bits:
            return bits  # 0...0
        flipped = "".join(["1" if b == "0" else "0" for b in bits])
        return Posit._increment_bits(flipped)


    @staticmethod
    def _increment_bits(bits: str) -> str:
        """Adds 1 to a bit string of arbitrary length, handling overflow."""
        n = len(bits)
        val = int(bits, 2) + 1
        bin_val = format(val, f"0{n}b")
        return bin_val[-n:]  # Return last n bits in case of overflow

