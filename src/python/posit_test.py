import unittest
from posit import Posit


class PositTest(unittest.TestCase):
    def test_posit_creation(self):
        p = Posit(10**27, n=8)
        self.assertEqual(p.to_float(), p.MAX_POS)
        p = Posit(3)
        self.assertEqual(p.to_float(), 3.0)
        p = Posit(0.5)
        self.assertEqual(p.to_float(), 0.5)
        p = Posit(-0.061, 64)
        self.assertEqual(p.to_float(), -0.061)


    def test_conversion_64(self):
        for i in range(-1000, 1000):
            v = i / 100000
            p = Posit(v, n=64)
            self.assertEqual(p.to_float(), v)


    def test_multiplication(self):
        nan = float('nan')
        inf = float('inf')
        test_cases = [
            (Posit(1.5), Posit(2.0), Posit(3.0)),      # Positive * Positive
            (Posit(-1.5), Posit(-2.0), Posit(3.0)),    # Negative * Negative
            (Posit(1.5), Posit(-2.0), Posit(-3.0)),    # Positive * Negative
            (Posit(0), Posit(2.0), Posit(0)),          # Zero * Positive
            (Posit(1.0), Posit(0), Posit(0)),          # Positive * Zero
            (Posit(1.0), Posit(inf), Posit(nan)),      # Positive * NaR
            (Posit(0.3), Posit(10), Posit(3, n=8)),    # Mixed precision
            (Posit(0.3), Posit(10), Posit(3, n=64)),   # Mixed precision
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b):
                result = a * b
                self.assertEqual(result, expected)


    def test_addition(self):
        nan = float('nan')
        inf = float('inf')
        test_cases = [
            (Posit(1.5), Posit(2.0), Posit(3.5)),      # Positive * Positive
            (Posit(-1.5), Posit(-2.0), Posit(-3.5)),    # Negative * Negative
            (Posit(1.5), Posit(-2.0), Posit(-0.5)),    # Positive * Negative
            (Posit(0), Posit(2.0), Posit(2.0)),          # Zero * Positive
            (Posit(1.0), Posit(0), Posit(1.0)),          # Positive * Zero
            (Posit(1.0), Posit(inf), Posit(nan)),      # Positive * NaR
        ]

        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b):
                result = a + b
                self.assertEqual(result, expected)
if __name__ == "__main__":
    unittest.main()
