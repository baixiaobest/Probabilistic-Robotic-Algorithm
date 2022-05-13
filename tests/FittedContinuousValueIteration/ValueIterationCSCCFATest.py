import unittest
import numpy as np
import src.OptimalControl.FittedContinuousValueIteration.ValueIterationCSCCFA as VI
import src.OptimalControl.FittedContinuousValueIteration.QuadraticFuncApproximator as QFA
import src.OptimalControl.FittedContinuousValueIteration.StateSpaceDynamics as SS
from src.OptimalControl.FittedContinuousValueIteration.common import get_policy_at
import matplotlib.pyplot as plt
from matplotlib import cm


def cost_to_go(x, u):
    '''
    x: state of size 2
    u: control of size 1
    '''
    return x@np.diag([1, 1])@x + u*u

def get_double_integrator_dynamics():
    # Dynamics of double integrator
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.identity(2)
    D = np.zeros((2, 1))
    double_integrator_ss = SS.StateSpaceDynamics(A, B, C, D)
    return double_integrator_ss

def compute_value_function():
    double_integrator_ss = get_double_integrator_dynamics()

    # Quadratic function approximation of value function
    func_approx = QFA.QuadraticFunctionApproximator(2)

    # Time step for each value iteration
    delta_t = 0.1
    total_num_iterations = 100

    # Sample in the statespace for fitting of function approximator
    q = np.linspace(-20, 20, 41)
    q_dt = np.linspace(-10, 10, 21)
    q_m, q_dt_m = np.meshgrid(q, q_dt)
    sample_states = []
    for row in range(q_m.shape[0]):
        for col in range(q_m.shape[1]):
            sample_states.append(np.array([q_m[row][col], q_dt_m[row][col]]))

    vi = VI.ValueIterationCSCSFA(sample_states, double_integrator_ss, cost_to_go, func_approx, delta_t)

    for i in range(total_num_iterations):
        vi.iterate()
        print(func_approx.get_A())

    '''
    Got approximated value function x'Ax with 
    A=
        [[1.83421586 1.09046314]
         [1.09046314 1.89109854]]
    '''

def visualize_value_function_and_policy():
    A = np.array([[1.83421586, 1.09046314],
                  [1.09046314, 1.89109854]])

    # Since we know value function J is quadratic of form x'Ax,
    # then its derivative is 2Ax.
    def dJdx(x):
        return 2*A@x

    ss = get_double_integrator_dynamics()

    # Generate state space mesh
    q = np.linspace(-20, 20, 81)
    q_dt = np.linspace(-10, 10, 41)
    q_m, q_dt_m = np.meshgrid(q, q_dt)
    value_func = np.zeros(q_m.shape)
    policy = np.zeros(q_m.shape)

    # Get value and policy on mesh grid
    for row in range(q_m.shape[0]):
        for col in range(q_m.shape[1]):
            x = np.array([q_m[row][col], q_dt_m[row][col]])
            value_func[row][col] = x@A@x
            policy[row][col] = get_policy_at(x, 0, cost_to_go, dJdx, ss.dxdt)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(q_m, q_dt_m, value_func, cmap=cm.coolwarm)
    ax.set_xlabel('q')
    ax.set_ylabel('q_dt')
    ax.set_title("value function")

    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax2.plot_surface(q_m, q_dt_m, policy, cmap=cm.coolwarm)
    ax2.set_xlabel('q')
    ax2.set_ylabel('q_dt')
    ax2.set_title("policy")

    plt.show()

if __name__ == '__main__':
    # compute_value_function()
    visualize_value_function_and_policy()

