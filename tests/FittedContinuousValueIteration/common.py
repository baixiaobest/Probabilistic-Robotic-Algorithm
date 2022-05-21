import numpy as np
import src.OptimalControl.FittedContinuousValueIteration.StateSpaceDynamics as SS

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