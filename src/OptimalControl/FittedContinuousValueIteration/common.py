from scipy.optimize import minimize

def get_policy_at(x, u0, cost_to_go, dJdx, dxdt, maxiter=10, control_bounds=None):
    '''
    x: State at which to generate the policy
    u: Initial guess of the control.
    cost_to_go: Cost to go function.
    dJdx: Partial derivative of value function with respect to x, dJdx(x).
    dxdt: Derivative of state with respect to time, dxdt(x).
    maxiter: number of iteration.
    The function calculate the policy given dynamics and value function derivative.
    It does π(x) = argmin_u[l(x,u) + dJdx * f(x,u)], where l(x,u) is cost to go,
    f(x,u) is the dynamics of the system.
    '''
    def cost_function(u):
        return cost_to_go(x, u) + dJdx(x)@dxdt(x, u)

    if control_bounds is None:
        method = "BFGS"
    else:
        method = "L-BFGS-B"

    res = minimize(cost_function, u0, method=method, options={'maxiter': maxiter, 'disp': False}, bounds=control_bounds)

    return res.x
