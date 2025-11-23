import math
import struct
from typing import override


class Posit:
    """
    Implementation of a Posit number.

    Implements parts of the [Posit standard](https://posithub.org/about)
    """

    def __init__(self, value: float|str = 0.0, n: int = 32):
        """
        Initialize a Posit number with the given value, precision, and exponent size.

        A posit is structured as follows:
        Sign Bit | Regime Bits | Exponent Bits | Fraction Bits
        n-1..n-2 |  n-3..n-k   |  n-k-1..n-m   |   n-m-1..0

        1. Sign Bit (1 bit): 
           0 for positive, 1 for negative. 
           Negative numbers are represented using Two's Complement.

        2. Regime Bits (Variable length):
           A run of identical bits terminated by an opposite bit. 
           Determines the regime value (k), representing a coarse scale factor of 16^k.

        3. Exponent Bits (Up to 2 bits):
           The next 2 bits represent the unsigned exponent (e) over the base 2.
           This field is truncated if the regime consumes too many bits.

        4. Fraction Bits (Remaining bits):
           The remaining bits form the fraction (f) of the significand 1.f.

        Args:
            value (float|str): The value of the Posit number,
                or the binary representation as a bit string.
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

        if isinstance(value, (float, int)):
            if value != 0:
                self._float_to_posit(value)
        elif isinstance(value, str):
            self._set_bits(value.ljust(n, '0'))
        else:
            raise ValueError(value + " is not a float or string!")


    def _set_bits(self, bits: str):
        if not set(bits).issubset({'0', '1'}):
            raise ValueError(f"Error: '{bits}' is not a binary string!")
        if len(bits) != self.BITWIDTH:
            raise ValueError(f"Expected {self.BITWIDTH} bits, got {len(bits)}")
        self._bits = bits


    def is_zero(self) -> bool:
        """Tests if the value of this posit is equal to 0.0"""
        return self._bits == '0' * self.BITWIDTH


    def is_nar(self) -> bool:
        """Tests if the value of this posit is not-a-real (NaR)"""
        return self._bits == '1' + '0' * (self.BITWIDTH - 1)


    def clone(self) -> 'Posit':
        """Returns a deep copy of this instance."""
        res = Posit(n=self.BITWIDTH)
        res._bits = "".join(c for c in self._bits)
        return res


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
        if s == 1:
            bits = self._twos_complement(bits)
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
        float_val = (1 + f) * 2**(4*r + e)
        return float_val if s == 0 else -float_val


    def __eq__(self, other: 'Posit') -> bool:
        """
        Tests if two posits are numerically equal.
        Handles posits of different precisions (BITWIDTH).
        """
        if not isinstance(other, Posit):
            return False

        # === Special Cases === #
        if self.is_nar() and other.is_nar():
            return True # Posit spec
        if self.is_zero() and other.is_zero():
            return True
        if self.is_nar() or other.is_nar():
            return False

        # === Compare Components === #
        s1, r1, e1, f1 = self._get_components()
        s2, r2, e2, f2 = other._get_components()

        if s1 != s2 or r1 != r2 or e1 != e2:
            return False

        # === Compare Fractions === #
        max_len = max(len(f1), len(f2)) # "1" (0.5) should equal "100" (0.500)
        return f1.ljust(max_len, '0') == f2.ljust(max_len, '0')



    def __neg__(self) -> 'Posit':
        res = Posit(0, n=self.BITWIDTH)
        res._bits = self._twos_complement(self._bits)
        return res


    def __add__(self, other: 'Posit') -> 'Posit':
        """
        Adds this Posit to another Posit.

        Uses a unified-mantissa addition, meaning, components get massaged into:
        P = Mantissa * 2^Scale

        The Addition logic aligns the scales to the smaller common scale to preserve precision.
        Given P1 = M1 * 2^S1 and P2 = M2 * 2^S2 where S1 > S2:
        P1 becomes (M1 << (S1 - S2)) * 2^S2
        Then P1 + P2 = ((M1 << (S1 - S2)) + M2) * 2^S2
        """
        if not isinstance(other, Posit):
            raise ValueError(f"__add__: Expected Posit, got {type(other)}")
        if self.BITWIDTH != other.BITWIDTH:
            raise ValueError(f"Cannot add {self.BITWIDTH}-bit Posit with {other.BITWIDTH}-bit Posit!")

        # --- Handle Special Cases ---
        if self.is_nar() or other.is_nar():
            return Posit(n=self.BITWIDTH, value=float('nan'))
        if self.is_zero():
            return other.clone()
        if other.is_zero():
            return self.clone()

        # --- Calculate components --- #
        s_self, r_self, e_self, f_self = self._get_components()
        s_other, r_other, e_other, f_other = other._get_components()

        # --- Calculate Mantissa and Scale --- #
        mantissa_self, scale_self = self._unified_mantissa(r_self, e_self, f_self)
        mantissa_other, scale_other = self._unified_mantissa(r_other, e_other, f_other)

        # === Align to smaller, common scale === #
        if scale_self > scale_other:
            mantissa_self <<= (scale_self - scale_other)
            scale = scale_other
        else:
            mantissa_other <<= (scale_other - scale_self)
            scale = scale_self

        # === Perform addition === #
        if s_self == s_other:  # Same sign: Add magnitudes
            mantissa = mantissa_self + mantissa_other
            s = s_self
        else:  # Different signs: Subtract magnitudes
            if mantissa_self > mantissa_other:
                mantissa = mantissa_self - mantissa_other
                s = s_self
            elif mantissa_other > mantissa_self:
                mantissa = mantissa_other - mantissa_self
                s = s_other
            else:  # Exact cancellation
                return Posit(n=self.BITWIDTH)

        # --- Transform unified mantissa back into posit components --- #
        r, e, f = self._posit_parts(mantissa, scale)

        res = Posit(0, n=self.BITWIDTH)
        res._encode_compnents(s, r, e, f)
        return res


    def __mul__(self, other: 'Posit') -> 'Posit':
        """
        Multiplies this Posit by another Posit.

        Uses a unified-mantissa multiplication, meaning, components get massaed into:
        P = Mantissa * 2^Scale

        The Multiplication can then be rewritten as:
        (Mantissa1 * 2^Scale1) * (Mantissa2 * 2^Scale2) = (Mantissa1 * Mantissa2) * 2^(Scale1 + Scale2)
        """
        if not isinstance(other, Posit):
            raise ValueError(f"__mul__: Expected Posit, got {type(other)}")
        if self.BITWIDTH != other.BITWIDTH:
            raise ValueError(f"Cannot multiply {self.BITWIDTH}-bit Posit with {other.BITWIDTH}-bit Posit!")

        # === Handle Special Cases === #
        if self.is_nar() or other.is_nar():
            return Posit(float('nan'), n=self.BITWIDTH)
        if self.is_zero() or other.is_zero():
            return Posit(0.0, n=self.BITWIDTH)

        # --- Calculate components --- #
        s_self, r_self, e_self, f_self = self._get_components()
        s_other, r_other, e_other, f_other = other._get_components()

        # --- Calculate Mantissa and Scale --- #
        mantissa_self, scale_self = self._unified_mantissa(r_self, e_self, f_self)
        mantissa_other, scale_other = self._unified_mantissa(r_other, e_other, f_other)

        # === Perform multiplication === #
        mantissa = mantissa_self * mantissa_other
        scale = scale_self + scale_other
        sign = 1 if s_self != s_other else 0

        # --- Transform unified mantissa back into posit components --- #
        r, e, f = self._posit_parts(mantissa, scale)

        res = Posit(0, n=self.BITWIDTH)
        res._encode_compnents(sign, r, e, f)
        return res


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


    @staticmethod
    def _unified_mantissa(r: int, e: int, f: str) -> tuple[int, int]:
        """
        Converts regime, exponent and fraction into mantissa and exponent:
        1.f * 2^(4r + e) -> m * 2^s

        While the fraction part has an implicit leading 1., the mantissa is
        entirely displayed and interpreted as an integer without a decimal
        point.

        Args:
            r (int): Regime value (numerical exponent of useed).
            e (int): Exponent value (0-3).
            f (str): Fraction bits.

        Returns:
            m, s (tuple[int, int]): Mantissa and scale (exponent). 
        """
        f_len = len(f)
        # Fraction currently is (1).XXX. Attach implicit one and interpret
        # as without decimal point (leftshift)
        m = (1 << f_len) | (int(f, 2) if f_len > 0 else 0)
        # Build unified mantissa (4r + e) and deduct leftshift of fraction
        s = (r * 4 + e) - f_len
        return m, s


    @staticmethod
    def _posit_parts(mantissa: int, scale: int) -> tuple[int, int, str]:
        """
        Converts a floating-point value composed of a mantissa and exponent
        into posit regime, exponent and fraction representation:
        m * 2^s -> 1.f * 2^(4r + e)

        While the mantissa is interpreted as an integer value, the fraction has
        an implicit 1 leading before the decimal point (1.f).

        Args:
            mantissa (int): An integer value
            scale (int): Integer exponent over the base 2.

        Returns:
            r, e, f (tuple[int, int, str]): Posit components (except sign)
        """
        mantissa_len = mantissa.bit_length() - 1  # Deduct implicit 1
        total_exponent = scale + mantissa_len  # Sum of scales + normalization shift
        r = total_exponent >> 2  # div 4
        e = total_exponent & 3   # mod 4
        f_int = mantissa & ((1 << mantissa_len) - 1)  # Discard leading 1
        f = format(f_int, f'0{mantissa_len}b') if mantissa_len > 0 else ""
        return r, e, f

