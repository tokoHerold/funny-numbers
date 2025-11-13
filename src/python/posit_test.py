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



if __name__ == "__main__":
    unittest.main()
