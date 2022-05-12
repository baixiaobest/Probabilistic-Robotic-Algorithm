from scipy.optimize import minimize
import numpy as np

'''
Value iteration continuous state continuous control with function approximator.
https://underactuated.mit.edu/dp.html#section3
'''
class ValueIterationCSCSFA:
    '''
    sample_states: List of sample states (np array).
    dynamics: Continuous dynamics of the system.
    cost_to_go: Cost to go function, parameters are states and controls.
    function_approximator: Function approximation of value function.
    delta_t: Sample time of the value iteration.
    '''
    def __init__(self, sample_states, dynamics, cost_to_go, function_approximator, delta_t):
        self.sample_states = sample_states
        self.dynamics = dynamics
        self.cost_to_go = cost_to_go
        self.function_approximator = function_approximator
        self.delta_t = delta_t
        # Optimal control at sample states, this is improved over iterations.
        self.optimal_controls = [np.zeros(dynamics.num_controls()) for i in range(len(sample_states))]
        self.control_optimizer_method = 'BFGS'
        self.control_optimizer_options = {'maxiter': 10, 'disp': False}
        self.func_approx_optimizer_method = 'BFGS'
        self.func_approx_optimizer_options = {'maxiter': 10, 'disp': False}

    def iterate(self, num_iteration=1):
        for i in range(num_iteration):
            J_desired = [0 for i in range(len(self.sample_states))]
            # Iterate over each sample states, calculate minimizing control and desired J.
            for idx, state in enumerate(self.sample_states):
                u = self.optimal_controls[idx]
                res = minimize(
                    self._control_cost_function,
                    u,
                    state,
                    method=self.control_optimizer_method,
                    options=self.control_optimizer_options)

                u_opt = res.x
                self.optimal_controls[idx] = u_opt
                J_desired[idx] = self._control_cost_function(u_opt, state)

            # Optimize the function approximator.
            alpha = self.function_approximator.get_parameters()
            res = minimize(
                self._func_approx_cost_function,
                alpha,
                J_desired,
                method=self.func_approx_optimizer_method,
                options=self.func_approx_optimizer_options)

            alpha_opt = res.x
            self.function_approximator.set_parameters(alpha_opt)

    def _control_cost_function(self, u, args):
        '''
        u: control input to the system.
        args: contains state x of the system. args is a tuple (x)
        '''
        x = args
        l = self.cost_to_go(x, u) * self.delta_t
        x_next = x + self.dynamics.dxdt(x, u) * self.delta_t
        value = self.function_approximator.value_at(x_next)
        return l + value

    def _func_approx_cost_function(self, alpha, args):
        '''
        alpha: parameters of the function approximator.
        args: contains desired J (J values at sample points).
        '''
        J_desired = args
        self.function_approximator.set_parameters(alpha)
        cost = 0
        for idx, x in enumerate(self.sample_states):
            cost += (self.function_approximator.value_at(x) - J_desired[idx]) ** 2
        return cost
