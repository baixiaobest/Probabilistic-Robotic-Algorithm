import numpy as np

""" Configurable state space cost to go. """
class CostToGo:
    """
    config: configuration of the state space. [{min, max, resolution}, {min, max, resolution}...]
    """
    def __init__(self, configs):
        self.dimensions = len(configs)
        self.configs = configs
        shape = ()
        for config in configs:
            size = int((config["max"] - config["min"]) / config["resolution"]) + 1
            shape += (size,)
        self.state_space_cost = np.zeros(shape)

    """ Given state, get the cost to go. """
    def get_cost(self, state):
        val = self.state_space_cost
        for i in range(len(state)):
            state_config = self.configs[i]
            if state[i] > state_config["max"] or state[i] < state_config["min"]:
                raise ValueError("input state not valid, not in the defined state_space")
            state_idx = int((state[i] - state_config["min"]) / state_config["resolution"])
            # This is equivalent to iteratively indexing: val[][]...[]
            val = val[state_idx]
        return val

    """ Set the cost to go at given state. """
    def set_cost(self, state, cost):
        val = self.state_space_cost
        for i in range(len(state)):
            state_config = self.configs[i]
            if state[i] > state_config["max"] or state[i] < state_config["min"]:
                raise ValueError("input state not valid, not in the defined state_space")
            state_idx = int((state[i] - state_config["min"]) / state_config["resolution"])
            # This is equivalent to iteratively indexing: val[][]...[] = cost
            if i == len(state) - 1:
                val[state_idx] = cost
            else:
                val = val[state_idx]

    def get_state_dimension(self):
        return len(self.configs)

    def get_state_space_configuration(self):
        return self.configs

    def get_number_discrete_state_values(self, dim):
        min = self.configs[dim]["min"]
        max = self.configs[dim]["max"]
        res = self.configs[dim]["resolution"]

        return int((max - min) / res)
