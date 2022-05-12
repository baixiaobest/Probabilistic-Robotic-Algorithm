import numpy as np

'''
This is a function approximation using f = x'Ax.
Where A is symmetric matrix. x is vector.
'''
class QuadraticFunctionApproximator:
    '''
    num_variables: For quadratic function x'Ax, the number of components in vector x.
    '''
    def __init__(self, num_variables=1):
        self.num_variables = num_variables
        self.A = np.identity(num_variables)
        self.parameters = np.zeros(int((1+num_variables) * self.num_variables/2))
        idx = 0
        for i in range(num_variables):
            for j in range(i, num_variables):
                self.parameters[idx] = self.A[i][j]
                idx += 1

    def get_parameters(self):
        return self.parameters

    def get_A(self):
        return self.A

    def set_parameters(self, p):
        if p.shape[0] != self.parameters.shape[0]:
            raise ValueError("set_parameters: Input argument length does not match the dimension of the parameters")
        self.parameters = p

        # Synchronize the parameter with the A matrix as well.
        idx = 0
        for i in range(self.num_variables):
            for j in range(i, self.num_variables):
                # A is symmetric
                self.A[i][j] = self.parameters[idx]
                self.A[j][i] = self.parameters[idx]
                idx += 1

    def value_at(self, x):
        return x@self.A@x