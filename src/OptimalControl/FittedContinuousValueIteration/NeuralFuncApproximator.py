import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class NeuralFuncApproximator:
    '''
    num_variables: size of state x in approximated functino f(x).
    hl
    '''
    def __init__(self, num_variables=1, hl1=10, hl2=10):
        self.num_variables = num_variables
        self.hl1 = hl1
        self.hl2 = hl2
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(num_variables, hl1)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hl1, hl2)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(hl2, 1)),
            ('relu3', nn.ReLU())]))

    def value_at(self, x):
        return self.model.forward(torch.tensor(x).float()).detach().numpy()

    def get_parameters(self):
        fc1_weight = self.model.fc1.weight.detach().numpy()
        fc1_weight = np.resize(fc1_weight, fc1_weight.shape[0] * fc1_weight.shape[1])
        fc2_weight = self.model.fc2.weight.detach().numpy()
        fc2_weight = np.resize(fc2_weight, fc2_weight.shape[0] * fc2_weight.shape[1])
        fc3_weight = self.model.fc3.weight.detach().numpy()
        fc3_weight = np.resize(fc3_weight, fc3_weight.shape[0] * fc3_weight.shape[1])
        fc1_bias = self.model.fc1.bias.detach().numpy()
        fc2_bias = self.model.fc2.bias.detach().numpy()
        fc3_bias = self.model.fc3.bias.detach().numpy()
        return np.concatenate((fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias))

    def set_parameters(self, param):
        fc1_weight_end = self.num_variables * self.hl1
        fc1_bias_end = fc1_weight_end + self.hl1
        fc2_weight_end = fc1_bias_end + self.hl2 * self.hl1
        fc2_bias_end = fc2_weight_end + self.hl2
        fc3_weight_end = fc2_bias_end + self.hl2
        fc3_bias_end = fc3_weight_end + 1

        fc1_weight = param[0:fc1_weight_end]
        fc1_weight = np.resize(fc1_weight, (self.hl1, self.num_variables))
        fc1_bias = param[fc1_weight_end:fc1_bias_end]

        fc2_weight = param[fc1_bias_end:fc2_weight_end]
        fc2_weight = np.resize(fc2_weight, (self.hl2, self.hl1))
        fc2_bias = param[fc2_weight_end:fc2_bias_end]

        fc3_weight = param[fc2_bias_end: fc3_weight_end]
        fc3_weight = np.resize(fc3_weight, (1, self.hl2))
        fc3_bias = param[fc3_weight_end:fc3_bias_end]

        state_dict = OrderedDict([
            ('fc1.weight', torch.tensor(fc1_weight).float()),
            ('fc1.bias', torch.tensor(fc1_bias).float()),
            ('fc2.weight', torch.tensor(fc2_weight).float()),
            ('fc2.bias', torch.tensor(fc2_bias).float()),
            ('fc3.weight', torch.tensor(fc3_weight).float()),
            ('fc3.bias', torch.tensor(fc3_bias).float())])

        self.model.load_state_dict(state_dict, strict=True)

    def print_parameters(self):
        for param in self.model.parameters():
            print(param)