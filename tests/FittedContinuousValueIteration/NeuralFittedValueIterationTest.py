import unittest
import numpy as np
import src.OptimalControl.FittedContinuousValueIteration.ValueIterationCSCCFA as VI
import src.OptimalControl.FittedContinuousValueIteration.NeuralFuncApproximator2 as NFA
from tests.FittedContinuousValueIteration.common import *
import src.OptimalControl.FittedContinuousValueIteration.StateSpaceDynamics as SS
from src.OptimalControl.FittedContinuousValueIteration.common import get_policy_at
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import odeint

def get_nn_parameters():
    return np.array(
        [-1.31108882e+00, - 4.31419593e-01,  4.79677550e-01,  6.48711621e-01,
         5.79829770e-01,  7.12409686e-01, -1.90525399e+00,  - 1.06601274e+00,
         7.70179397e-01,  1.13345440e+00,  4.89613686e-01,  8.70417444e-01,
         - 2.58314395e+00, - 2.08568321e+00,  6.49644327e-01,  1.06554774e+00,
         3.83310229e-01,  5.13937561e-01,  7.38398618e-01,  1.14849890e+00,
         1.55589130e-01,  6.71134197e-01,  5.31070251e-01,  3.26836595e-01,
         6.70165622e-01,  9.55753867e-01,  6.12714610e-01,  2.40441009e-01,
         9.31969385e-01,  7.72941368e-01,  1.09720140e+00,  1.70897389e-01,
         4.22811099e-02,  1.26067585e+00,  2.83383131e-01,  6.49195151e-01,
         1.63073772e+00,  8.65286189e-01,  2.99243971e-01,  5.20835661e-01,
         2.66813981e-01,  1.92345837e-01,  2.38527046e-01,  4.83457649e-01,
         8.66686061e-01,  9.45317691e-01,  1.21737745e+00,  5.13878831e-01,
         2.08108391e-01,  6.45281163e-01,  9.10915932e-01,  9.20899969e-01,
         8.78242222e-01,  5.65058553e-01,  6.72126984e-01,  8.02615679e-01,
         1.62470423e+00, - 2.96548324e-03,  1.07478723e-01,  1.75399263e-01,
         2.69964351e-01,  4.58819344e-01,  5.56448079e-01,  9.46280220e-01,
         3.55498800e-01,  5.29908283e-01,  1.07238054e+00,  4.00830960e-02,
         6.29132871e-01,  1.05986504e-01,  2.79196745e-01,  8.89090753e-01,
         3.67183861e-01,  1.35038323e+00,  2.51016086e-01,  1.24017884e-01,
         1.11208471e+00,  4.86638522e-01,  7.49632119e-01,  4.71695831e-01,
         8.99412728e-01,  6.84630035e-01,  8.63502378e-01,  1.11250955e+00,
         7.48766723e-01,  2.99567376e-02,  1.29739942e+00,  8.53667271e-01,
         2.92164970e-01,  1.03348347e-01,  8.43444682e-01,  7.56669908e-01,
         2.66730994e-01,  6.38711807e-01,  7.23012585e-01,  5.14485185e-01,
         1.38116418e+00, -1.80037708e-03,  1.11127279e-01,  1.19860087e-01,
         8.23148810e-01,  7.15495164e-01,  2.40796660e-01,  1.18956989e+00,
         5.54998669e-02,  2.09800771e-01,  1.27663882e+00, - 1.67958375e-02,
         3.17370026e-01,  2.52944908e-01,  3.28314110e-01,  8.11036100e-01,
         2.56184516e-01,  5.07018306e-01,  7.76235905e-03,  9.11236487e-01,
         7.03849180e-01,  9.49076470e-01,  9.24269941e-01,  5.56679371e-01,
         7.70494427e-01,  1.85767232e+00,  1.55262243e+00,  1.51736681e+00,
         1.43314061e+00,  1.55943979e+00,  1.31707626e+00,  1.46938070e+00,
         1.23904417e+00,  6.46099067e-01,  1.97690907e-01,  6.61614043e-01,
         5.34544020e-01,  7.64015638e-01,  1.56939856e-01,  6.07930858e-02,
         1.03258594e+00,  4.63884165e-01,  8.72023597e-01,  5.60285629e-01,
         4.90558354e-01,  4.98268739e-01,  1.15535588e-01,  8.47802304e-01,
         3.40676669e-01,  8.36485606e-01,  1.70593694e-01,  8.01138777e-01,
         8.12986415e-01,  1.65574555e-02,  2.31358882e-01])

def get_dJdx(func_approx):
    def dJdx(x):
        epsilon = 1e-6
        djdx = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            x_hat = np.copy(x)
            x_hat[i] = x_hat[i] + epsilon
            djdx[i] = (func_approx.value_at(x_hat) - func_approx.value_at(x)) / epsilon
        return djdx
    return dJdx

def compute_value_function():
    '''
    A grid of sample states is used for the computation of the value function.
    For each state at each iteration, an optimal policy and value will be computed.
    The value function is approximated as an quadratic function of x, or J(x) = x'Ax.
    '''
    double_integrator_ss = get_double_integrator_dynamics()

    # Quadratic function approximation of value function
    func_approx = NFA.NeuralFuncApproximator2(2, hl1=10, hl2=10)
    # func_approx.set_parameters(get_nn_parameters())

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


    vi = VI.ValueIterationCSCSFA(
        sample_states, double_integrator_ss, cost_to_go, func_approx, delta_t, method='CG', func_approx_maxiter=1)

    print("Initial parameters")
    print(func_approx.get_parameters())

    for i in range(total_num_iterations):
        vi.iterate()
        print(f"interation: {i}")
        print(func_approx.get_parameters())

    print("Optimized parameters")
    print(func_approx.get_parameters())

def visualize_value_function_and_policy():
    """
    Draw the graph of value function and the policy.
    """
    ss = get_double_integrator_dynamics()

    func_approx = NFA.NeuralFuncApproximator2(2, hl1=10, hl2=10)
    func_approx.set_parameters(get_nn_parameters())

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
            value_func[row][col] = func_approx.value_at(x)
            policy[row][col] = get_policy_at(x, 0, cost_to_go, get_dJdx(func_approx), ss.dxdt)

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

def visualize_control():
    """
    Given an initial condition of the system, try to control the system.
    """
    x0 = np.array([10, 20])

    ss = get_double_integrator_dynamics()
    func_approx = NFA.NeuralFuncApproximator2(2, hl1=10, hl2=10)
    func_approx.set_parameters(get_nn_parameters())

    dJdx = get_dJdx(func_approx)

    def controlled_dynamics(x, args):
        u = get_policy_at(x, 0, cost_to_go, dJdx, ss.dxdt, maxiter=10)
        return ss.dxdt(x, u)

    t = np.linspace(0, 50, 1000)
    xt = odeint(controlled_dynamics, x0, t)

    fig1 = plt.figure()
    ax = plt.gca()
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    plt.plot(t, xt[:, 0])

    fig2 = plt.figure()
    ax = plt.gca()
    ax.set_xlabel("Time")
    ax.set_ylabel("Velocity")
    plt.plot(t, xt[:, 1])

    plt.show()

if __name__ == '__main__':
    # compute_value_function()
    visualize_value_function_and_policy()
    visualize_control()

