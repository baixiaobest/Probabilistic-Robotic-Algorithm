import unittest
import src.OptimalControl.FittedContinuousValueIteration.NeuralFuncApproximator2 as nfa
import numpy as np

class NeuralFunctionApproximatorTest2(unittest.TestCase):
    def setUp(self):
        self.approx = nfa.NeuralFuncApproximator2(2, 10, 12)

    def test_value(self):
        x = np.array([10, 20.5])
        self.approx.value_at(x)
        param = self.approx.get_parameters()
        self.approx.set_parameters(param)

if __name__ == '__main__':
    unittest.main()