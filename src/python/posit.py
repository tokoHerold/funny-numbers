import math
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

        self._set_bits(self._float_to_posit(value))


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


    def _float_to_posit(self, value: float) -> str:
        """Converts a float into a n-bit posit"""
        n = self.BITWIDTH

        # Special case #1: zero
        if value == 0:
            return "0" * n
        # Special case #2: nar
        if math.isnan(value) or math.isinf(value):
            return "1" + "0" * (n - 1)

        # General case
        sign_bit = False  # Positive
        if value < 0:  # Check if negative
            sign_bit = True
            value = -value

        # Clamp to MIN/MAX_POS
        if value >= self.MAX_POS:
            posit_bits: str = '0' + '1' * (n - 1)
        elif value <= self.MIN_POS:
            posit_bits = '0' * (n - 1) + '1'
        else:  # --- General Case --- #
            # Find regime exponent k so that USEED**k <= value < USEED**(k+1)
            k = int(math.floor(math.log(value, self._USEED)))
            # Find exponent e so that the remainder `value / (useed**k) * 2**e` can be denoted as 1.XXX
            value = value / (self._USEED ** k)
            e = 0
            if value > 2:
                e = min(
                    int(math.floor(float(math.log2(value)))),
                    3,  # Defined in posit standard 2.0, max value of 2 bits
                )
            # The remainder is the fraction value, with an implicit 1.fffff... 
            f_val = value / (2.0**e) - 1.0

            # --- Build bit string --- #
            posit_bits = '0'  # Sign bit, positive for now
            remaining = n - 1
            # Build Regime
            if k >= 0:
                run_len = k + 1
                regime_base: str = '1' * run_len + '0'
            else: # k < 0
                run_len = abs(k)
                regime_base = '0' * run_len + '1'
            posit_bits += (regime_base[:remaining])  # Add regime bits
            remaining = n - len(posit_bits)
            assert(remaining >= 0)
            # Build exponent
            if remaining > 0:
                exp_bits = min(remaining, 2)  # Posit standard, es=2
                posit_bits += format(e, f'02b')[:exp_bits]
                remaining -= exp_bits
            # Build fraction
            for _ in range(remaining + 1):  # Use 1 guard bit for posit rounding
                f_val *= 2
                if f_val >= 1:
                    posit_bits += '1'
                    f_val -= 1
                elif f_val == 0.0:
                    break
                else:
                    posit_bits += '0'
            sticky = f_val > 0  # Determines whether posit value is inexact

            # --- Round value to nearest posit of size n --- #
            if len(posit_bits) < n + 1:  # n bits + 1 guard bit
                posit_bits = posit_bits.ljust(n, '0')  # value is exact
            else:  # = posit rounding required = #
                # Posit rounding works as follows:
                # let v be the n+1 bit posit currently stored in `posit_bits`
                # let u, w be the n bit posits, whose value surrounds v.
                # Since regime/exponents are the same, the bit strings
                # order works like for usigned integers here.
                u = posit_bits[:n]  # adjacent smaller n bit posit
                guard_bit = posit_bits[-1] # (n+1)th bit
                if guard_bit == '1' and (sticky or u[-1] == '1'):
                    # (n+1th) bit is not zero -> value in upper half/midpoint
                    # If fraction was truncated (sticky) -> round up
                    # Else value is exactly at midpoint:
                    #    LSB of u is 1 -> round up to next 'even' posit (spec)
                    posit_bits = self._increment_bits(u) # round up to v
                else:  # (n+1th) bit is -> value in lower half 
                    posit_bits = u # round down

        if sign_bit:  # If negative
            return self._twos_complement(posit_bits)
        return posit_bits


    def to_float(self) -> float:
        """Converts the value of this posit into a IEEE 754 float"""
        if self.is_zero():
            return +0.0
        if self.is_nar():
            return float('nan')

        n = self.BITWIDTH
        bits = self._bits
        # Read sign bit (s)
        s = 0 if self._bits[0] == '0' else 1
        # Parse regime (r)
        k = bits[1:].find('0' if bits[1] == '1' else '1')  # find run length
        if k == -1: # Regime is whole posit
            k = n-1
        r = -k if bits[1] == '0' else k - 1
        # Read exponent (e)
        remaining = bits[2 + k:] # Stored in next 2 bits (spec)
        e = int(remaining[:2].ljust(2, '0'), 2)  # consider truncated exponent
        # Calculate fraction (f)
        f = 0.0
        remaining = remaining[2:]
        if len(remaining) > 0:
            f = int(remaining.ljust(1, '0'), 2) / 2.0**len(remaining)

        # Calculate float value
        return ((1 - 3 * s) + f) * 2.0**((1 - 2 * s) * (4 * r + e + s))


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

    #     # Find e (exponent)
    #     remaining = value / (self._useed**self._k)
    #     self._e = (
    #         int(math.floor(math.log(remaining, self._useed))) if remaining >= 2 else 0
    #     )

    #     # Find f (fraction)
    #     remaining = remaining / (self._useed**self._e)
    #     self._f = remaining - 1

    #     def _to_float(self) -> float:
    #         """Converts the posit into the IEEE 754 floating point representation"""

    #         # Check if the posit is 0 or NAR
    #         if self._bits[1:] == "0" * (len(self._bits) - 1):
    #             if self._bits[0] == "0":
    #                 return +0.0  # All bits zero -> 0.0
    #             else:
    #                 return float("nan")  # First bit 1, rest 0 -> NAR

    #         # Build floating-point number
    #         s = 0 if self._bits[0] == "0.0" else 1

    #         # If self._k
