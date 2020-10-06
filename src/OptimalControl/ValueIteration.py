import numpy as np
import copy as cp
import src.OptimalControl.CostToGo as ctg

class ValueIteration:
    """
    dynamics: Dynamics of the system, contains update(state, u, delta_t) that integrate the dynamics.
    cost_to_go: Data structure that contains the cost to go of the state space.
    control_set: List of control vectors u that can be applied to the system.
    cost_function: A function that takes state and control vector and returns the cost.
    delta_t: State is updated by this time step during each iteration.
    discount_factor: Discount the cost to prevent cost value going to infinity.
    """
    def __init__(self, dynamics, cost_to_go, control_set, cost_function, delta_t, discount_factor=0.9):
        self.dynamics = dynamics
        self.cost_to_go = cost_to_go
        self.control_policy = ctg.CostToGo(cost_to_go.get_state_space_configuration())
        self.control_set = control_set
        self.cost_function = cost_function
        self.delta_t = delta_t
        self.discount_factor = discount_factor

    def value_iteration(self, num_iteration):
        dimension = self.cost_to_go.get_state_dimension()
        configs = self.cost_to_go.get_state_space_configuration()

        for iteration in range(num_iteration):
            print("Iteration: {0}".format(iteration + 1))
            new_cost_to_go = cp.deepcopy(self.cost_to_go)
            # Recursively update all the cells in the cost to go table.
            self._recursive_cost_iteration(np.zeros(dimension).astype(int), 0, self._get_value_update_func(new_cost_to_go))
            self.cost_to_go = new_cost_to_go

        # Generate control policy based on calculated cost.
        self.compute_control_policy()

    def compute_control_policy(self):
        dimension = self.cost_to_go.get_state_dimension()
        self._recursive_cost_iteration(np.zeros(dimension).astype(int), 0, self._generate_control_policy)

    def get_cost_to_go(self):
        return self.cost_to_go

    def get_policy(self):
        return self.control_policy

    """ 
    Iterate through all the states in cost_to_go and update the cost value.
    This is recursive operation because the dimension of the state is configurable.
    """
    def _recursive_cost_iteration(self, indices, dim, update_function):
        # Base case, we reach last dimension.
        if dim == len(indices):
            update_function(indices)
            return

        configs = self.cost_to_go.get_state_space_configuration()
        N = self.cost_to_go.get_number_discrete_state_values(dim)
        new_indices = np.array(indices, copy=True)

        # Iteratively go through all the states in dim'th dimension.
        for i in range(N):
            new_indices[dim] = i
            self._recursive_cost_iteration(new_indices, dim + 1, update_function)

    def _generate_control_policy(self, indices):
        state = self.cost_to_go.get_state_from_indices(indices)
        min_cost = np.Inf
        best_control = 0
        # Find the control action that minimizes the cost
        for control in self.control_set:
            cost_of_action = self.cost_function(state, control) * self.delta_t
            next_state = self.dynamics.update(state, control, self.delta_t)
            try:
                next_state_cost_to_go = self.cost_to_go.get_cost(next_state)
            # Next state is out of bound, cost could not be calculated.
            except ValueError:
                next_state_cost_to_go = np.Inf
            total_cost = cost_of_action + self.discount_factor * next_state_cost_to_go
            if total_cost < min_cost:
                min_cost = total_cost
                best_control = control

        self.control_policy.set_cost_by_indices(indices, best_control)

    def _get_value_update_func(self, new_cost_to_go):
        """ Perform value/cost update on state. """
        def _value_update(indices):
            min_cost = np.Inf
            state = new_cost_to_go.get_state_from_indices(indices)
            # Find the control action that minimizes the cost
            for u in self.control_set:
                cost_of_action = self.cost_function(state, u) * self.delta_t
                next_state = self.dynamics.update(state, u, self.delta_t)
                try:
                    next_state_cost_to_go = self.cost_to_go.get_cost(next_state)
                # Next state is out of bound, cost could not be calculated.
                except ValueError:
                    next_state_cost_to_go = np.Inf
                total_cost = cost_of_action + self.discount_factor * next_state_cost_to_go
                if total_cost < min_cost:
                    min_cost = total_cost

            if not min_cost == np.Inf:
                new_cost_to_go.set_cost_by_indices(indices, min_cost)
            # This case happens when the dynamics bring the state out of boundary of the cost to go table.
            # We can only approximate the cost.
            else:
                cost_of_inaction = self.cost_function(state, 0) * self.delta_t
                new_cost = self.discount_factor * self.cost_to_go.get_cost(state) + cost_of_inaction
                new_cost_to_go.set_cost_by_indices(indices, new_cost)

        return _value_update
