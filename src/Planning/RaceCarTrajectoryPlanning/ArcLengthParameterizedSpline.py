import scipy.integrate as integrate
import numpy as np
import numpy.linalg as la
import src.Planning.RaceCarTrajectoryPlanning.SplineFit as sf

'''
This class re-parameterizes a given list of splines by arc length of the splines.
It is based on "Arc-Length Parameterized Spline Curves for Real-Time Simulation".
'''
class ArcLengthParameterizedSplines:
    def __init__(self, dim=2, epsilon=0.001):
        '''
        :param dim: Dimension of the spline.
        :param epsilon: Accuracy when calculating the position of the spline given the length.
        '''
        self.splines = []
        self.spline_lengths = []
        self.dimension = dim
        self.epsilon = epsilon

    def get_splines(self):
        return self.splines

    def get_spline_lengths(self):
        return self.spline_lengths

    def add_spline(self, spline):
        '''
        :param spline: list of parameters [a, b, c, d] of spline.
            a,b,c,d can be np vector to represent higher dimensional spline.
            x(t) = a[0]*t^3 + b[0]*t^2 + c[0]*t + d[0]
            y(t) = a[1]*t^3 + b[1]*t^2 + c[1]*t + d[1]
            ...
            t is in range [0, 1]
        :return:
        '''
        self.splines.append(spline)
        self.spline_lengths.append(self._compute_spline_arc_length(spline))

    def _compute_spline_arc_length(self, spline):
        '''
        Given spline, calculate its length with parameter ranging from 0 to 1.
        :param spline: spline of list [a b c d].
        :return: arc length.
        '''
        def spline_derivative_func(t):
            sum = 0
            a = spline[0]
            b = spline[1]
            c = spline[2]
            for i in range(self.dimension):
                sum += (3 * a[i] * t ** 2 + 2 * b[i] * t + c[i]) ** 2

            return np.sqrt(sum)

        return integrate.quad(spline_derivative_func, 0, 1)[0]

    def _find_position_in_splines(self, length):
        '''
        Starting from the beginning of the splines, find the position of the
        point after traversing the given arc length.
        :param length: Given traversed lengths.
        :return: (position, tangent, t parameter, spline index)
        '''

        # Find which segment of splines the given length falls into.
        prev_seg_length_sum = 0
        seg_idx = 0
        while seg_idx < len(self.spline_lengths) and prev_seg_length_sum + self.spline_lengths[seg_idx] < length:
            prev_seg_length_sum += self.spline_lengths[seg_idx]
            seg_idx += 1

        if seg_idx == len(self.spline_lengths):
            seg_idx = len(self.spline_lengths) - 1

        length_curr_seg = length - prev_seg_length_sum
        spline = self.splines[seg_idx]

        pos, tang, t = self._get_position_in_spline(spline, length_curr_seg)

        return pos, tang, t, seg_idx

    def _get_position_in_spline(self, spline, length):
        '''
        Starting from the beginning of the spline, traverse spline with the given length,
        return the position after the traversal.
        :param spline: Array of [a, b, c, d] parameters, each element can be numpy vector.
        :param length: Length to traverse. Must be within t = [0, 1], where t parameterize the spline.
        :return: (Position, tangent, t parameter)
        '''
        a = spline[0]
        b = spline[1]
        c = spline[2]
        d = spline[3]
        l = 0
        r = 1
        mid = (l+r) / 2
        def spline_derivative(t):
            sum = 0
            for i in range(self.dimension):
                sum += (3 * a[i] * t ** 2 + 2 * b[i] * t + c[i]) ** 2
            return np.sqrt(sum)

        while True:
            length_mid = integrate.quad(spline_derivative, 0, mid)[0]
            if np.abs(length_mid-length) < self.epsilon:
                break
            if length < length_mid:
                r = mid
            else:
                l = mid
            mid = (l + r) / 2

        t = mid
        return a * t**3 + b * t**2 + c * t + d, \
               3 * a * t**2 + 2 * b * t + c,\
               t

    def compute_arc_length_parameterized_spline(self, num_seg):
        '''
        :param num_seg: Number of new spline segments.
        :return: list of arc length parameterized splines,
                 segment arc lengths,
                 parameter t at segment points,
                 spline indices in segment points
        '''
        total_length = sum(self.spline_lengths)
        seg_length = total_length / num_seg
        seg_points = []
        tangents = []
        ts = []
        indices = []

        # traverse the splines with fixed arc lengths,
        # and compute the position and tangent at each step.
        for i in range(0, num_seg + 1):
            traversed_length = i * seg_length
            pos, tang, t, idx = self._find_position_in_splines(traversed_length)
            seg_points.append(pos)
            tangents.append(tang / la.norm(tang))
            ts.append(t)
            indices.append(idx)

        arc_splines = []
        for i in range(len(seg_points) - 1):
            start_point = seg_points[i]
            end_point = seg_points[i+1]
            start_tangent = tangents[i]
            end_tangent = tangents[i+1]

            arc_splines.append(
                sf.EndpointsSplineFit(start_point, end_point, start_tangent, end_tangent, seg_length, self.dimension))

        return arc_splines, seg_length, ts, indices
