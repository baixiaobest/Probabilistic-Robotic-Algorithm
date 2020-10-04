import numpy as np
import copy as cp

class ValueIteration:
    """
    dynamics: Dynamics of the system, contains update(state, u, delta_t) that integrate the dynamics.
    cost_to_go: Data structure that contains the cost to go of the state space.
    control_set: List of control vectors u that can be applied to the system.
    cost_function: A function that takes state and control vector and returns the cost.
    """
    def __init__(self, dynamics, cost_to_go, control_set, cost_function, delta_t):
        self.dynamics = dynamics
        self.cost_to_go = cost_to_go
        self.control_set = control_set
        self.cost_function = cost_function
        self.delta_t = delta_t

    def value_iteration(self, num_iteration):
        for iteration in range(num_iteration):
            dimension = self.cost_to_go.get_dimension()
            configs = self.cost_to_go.get_state_space_configuration()
            # state is initialized to be the minimum of all states in cost to go
            # (in terms of generalized inequality in positive orthant).
            state = np.zeros(dimension)
            for dim in range(dimension):
                state[dim] = configs[dim]["min"]
            new_cost_to_go = cp.deepcopy(self.cost_to_go)
            self._recursive_cost_update(state, 0, new_cost_to_go)
            self.cost_to_go = new_cost_to_go

    def _recursive_cost_update(self, state, dim, new_cost_to_go):
        if dim == len(state):
            self._value_update(state, new_cost_to_go)

        configs = self.cost_to_go.get_state_space_configuration()
        N = self.cost_to_go.get_number_discrete_state_values(dim)
        new_state = np.array(state, copy=True)

        # Iteratively go through all the states in dim'th dimension.
        for i in range(N):
            val = configs[dim]["min"] + configs[dim]["resolution"] * i
            new_state[dim] = val
            self._recursive_cost_update(new_state, dim+1, new_cost_to_go)

    def _value_update(self, state, new_cost_to_go):
        min_cost = np.Inf
        for u in self.control_set:
            cost_of_action = self.cost_function(state, u)
            next_state = self.dynamics.update(state, u, self.delta_t)
            next_state_cost_to_go = self.cost_to_go.get_cost(next_state)
            total_cost = cost_of_action + next_state_cost_to_go
            if total_cost < min_cost:
                min_cost = total_cost

        if not min_cost == np.Inf:
            new_cost_to_go.set_cost(state, min_cost)
