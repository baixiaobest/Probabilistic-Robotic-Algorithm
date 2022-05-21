import unittest
import src.OptimalControl.FittedContinuousValueIteration.QuadraticFuncApproximator as qfa
import numpy as np

class QuadraticFunctionApproximatorTest(unittest.TestCase):
    def test_param(self):
        approx = qfa.QuadraticFunctionApproximator(3)
        p = approx.get_parameters()

        self.assertEqual(np.array_equal(p, np.array([1, 0, 0, 1, 0, 1])), True)

    def test_value(self):
        approx = qfa.QuadraticFunctionApproximator(3)
        p = np.array([1, 0, 0, 10, 0, 100])
        approx.set_parameters(p)
        x = np.array([2, 3, 4])
        v = approx.value_at(x)
        self.assertEqual(np.array_equal(v, 1694), True)

if __name__ == '__main__':
    unittest.main()
