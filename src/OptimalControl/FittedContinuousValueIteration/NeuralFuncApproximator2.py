import numpy as np
from numpy.random import random_sample

class NeuralFuncApproximator2:
    '''
    num_variables: size of state x in approximated functino f(x).
    hl
    '''
    def __init__(self, num_variables=1, hl1=10, hl2=10):
        self.num_variables = num_variables
        self.hl1 = hl1
        self.hl2 = hl2
        self.weight = [random_sample(num_variables*hl1), random_sample(hl1*hl2), random_sample(hl2)]
        self.bias = [random_sample(hl1), random_sample(hl2), random_sample(1)]
        self._update_weight_matrix()

    def _update_weight_matrix(self):
        self.A1 = np.resize(self.weight[0], (self.hl1, self.num_variables))
        self.A2 = np.resize(self.weight[1], (self.hl2, self.hl1))
        self.A3 = np.resize(self.weight[2], (1, self.hl2))
        self.b1 = self.bias[0]
        self.b2 = self.bias[1]
        self.b3 = self.bias[2]

    def _ReLU(self, v):
        return np.array([float(i>0) for i in v]) * v

    def value_at(self, x):
        # Hidden layer 1
        v = self._ReLU(self.A1@x + self.b1)
        # Hidden layer 2
        v = self._ReLU(self.A2@v + self.b2)
        # Output layer
        v = self.A3@v + self.b3
        return v[0]


    def get_parameters(self):
        param = np.concatenate(
            (self.weight[0], self.weight[1], self.weight[2], self.bias[0], self.bias[1], self.bias[2]))
        return param

    def set_parameters(self, param):
        weight1_end = self.num_variables * self.hl1
        weight2_end = weight1_end + self.hl1 * self.hl2
        weight3_end = weight2_end + self.hl2
        bias1_end = weight3_end + self.hl1
        bias2_end = bias1_end + self.hl2
        bias3_end = bias2_end + 1

        self.weight[0] = param[0:weight1_end]
        self.weight[1] = param[weight1_end:weight2_end]
        self.weight[2] = param[weight2_end:weight3_end]
        self.bias[0] = param[weight3_end:bias1_end]
        self.bias[1] = param[bias1_end:bias2_end]
        self.bias[2] = param[bias2_end:bias3_end]

        self._update_weight_matrix()