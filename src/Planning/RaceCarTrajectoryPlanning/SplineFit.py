import numpy as np
import numpy.linalg as la

def EndpointsSplineFit(start_point, end_point, start_tangent, end_tangent, end_t, dimension=2):
    '''
    Given start point and endpoint position and tangent, try to fit a spline
    with parameter ranging from 0 to end_t
    :param start_point: Start point position
    :param end_point: End point position
    :param start_tangent: Start point tangent
    :param end_tangent: End point tangent
    :param end_t: Ending parameter.
    :param dim: Number of dimensions of the curve.
    :return: Spline parameterized by vectors (numpy array) a, b, c, d, in the format
        of [a, b, c, d]. Spline is defined by Q(t) = at^3 + bt^2 + ct + d.
    '''
    # Spline parameters
    pa = np.zeros(dimension)
    pb = np.zeros(dimension)
    pc = np.zeros(dimension)
    pd = np.zeros(dimension)
    # For each dimension, Ax=b needs to be solved to obtain parameters.
    for dim in range(dimension):
        pc[dim] = start_tangent[dim]
        pd[dim] = start_point[dim]
        b = np.array(
            [end_point[dim] - start_tangent[dim] * end_t - start_point[dim],
             end_tangent[dim] - start_tangent[dim]])

        A = np.array([[end_t ** 3, end_t ** 2],
                      [3 * end_t ** 2, 2 * end_t]])

        ab = la.solve(A, b)
        pa[dim] = ab[0]
        pb[dim] = ab[1]

    return[pa, pb, pc, pd]