from scipy.optimize import minimize

def get_policy_at(x, u0, cost_to_go, dJdx, dxdt, mxiter=10):
    '''
    x: State at which to generate the policy
    u: Initial guess of the control.
    cost_to_go: Cost to go function.
    dJdx: Partial derivative of value function with respect to x, dJdx(x).
    dxdt: Derivative of state with respect to time, dxdt(x).
    mxiter: number of iteration.
    The function calculate the policy given dynamics and value function derivative.
    It does π(x) = argmin_u[l(x,u) + dJdx * f(x,u)], where l(x,u) is cost to go,
    f(x,u) is the dynamics of the system.
    '''
    def cost_function(u):
        return cost_to_go(x, u) + dJdx(x)@dxdt(x, u)

    res = minimize(cost_function, u0, method="BFGS", options={'maxiter': mxiter, 'disp': False})

    return res.x
