from scipy.optimize import shgo
import numpy as np
from numpy.linalg import norm


class VectorCubicSpline:
    """ a0, a1, a2, a3 are numpy vectors, they form the spline a0 + a1*s + a2*s^2 + a3*s^3 """
    def __init__(self, a0, a1, a2, a3):
        self.a0 = np.array(a0)
        self.a1 = np.array(a1)
        self.a2 = np.array(a2)
        self.a3 = np.array(a3)

    """ Get point given parameter s. """
    def get_point(self, s):
        return self.a0 + self.a1 * s + self.a2 * s**2 + self.a3 * s**3

    """ Get closest distance and parameter s to a point. 
        return (s, distance, point)"""
    def get_s_distance(self, point):
        def objective(s):
            point_on_spline = self.get_point(s)
            return norm(point_on_spline - point)

        bound = [(0, 1.0)]
        res = shgo(objective, bound)
        return res.x, objective(res.x)

    """ Get velocity on spline. Derivative of spline with respect to s. """
    def get_velocity(self, s):
        return self.a1 + 2 * self.a2 * s + 3 * self.a3 * s**2


""" Construct a spline by specifying start point, start point velocity, end point, endpoint velocity. """
def create_spline_start_end_point_velocity(start, start_vel, end, end_vel):
    start = np.array(start)
    start_vel = np.array(start_vel)
    end = np.array(end)
    end_vel = np.array(end_vel)

    a0 = start
    a1 = start_vel
    a2 = -3 * start + 3 * end - 2 * start_vel - end_vel
    a3 = 2 * start - 2 * end + start_vel + end_vel
    return VectorCubicSpline(a0, a1, a2, a3)

