import unittest
import src.OptimalControl.FittedContinuousValueIteration.NeuralFuncApproximator as nfa
import numpy as np

class NeuralFunctionApproximatorTest(unittest.TestCase):
    def setUp(self):
        self.approx = nfa.NeuralFuncApproximator(2, 10, 12)

    def test_value(self):
        x = np.array([10, 20.5])
        self.approx.value_at(x)
        self.approx.print_parameters()
        param = self.approx.get_parameters()
        self.approx.set_parameters(param)

if __name__ == '__main__':
    unittest.main()