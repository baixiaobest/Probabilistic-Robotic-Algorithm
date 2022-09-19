import scipy.integrate as integrate
import numpy as np
import numpy.linalg as la


class ArcLengthParameterizedSplines:
    def __init__(self, dim=2, epsilon=0.01):
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
                sum += 3 * a[i] * t ** 2 + 2 * b[i] * t + c[i]

            return np.sqrt(sum)

        return integrate.quad(spline_derivative_func, 0, 1).y

    def _find_position_from_length(self, length):
        '''
        Starting from the beginning of the splines, find the position of the
        point after traversing the given arc length.
        :param length: Given traversed lengths.
        :return: (position, tangent)
        '''
        prev_seg_length_sum = 0
        seg_idx = 0
        for i in range(len(self.spline_lengths)):
            if prev_seg_length_sum <= length \
                and prev_seg_length_sum + self.spline_lengths[i] > length:
                seg_idx = i
                break
            prev_seg_length_sum += self.spline_lengths[i]

        length_curr_seg = length - prev_seg_length_sum
        spline = self.splines[seg_idx]

        return self._get_seg_position_from_length(spline, length_curr_seg)

    def _get_seg_position_from_length(self, spline, length):
        '''
        Starting from the beginning of the spline, traverse spline with the given length,
        return the position after the traversal.
        :param spline: Array of [a, b, c, d] parameters, each element can be numpy vector.
        :param length: Length to traverse. Must be within t = [0, 1], where t parameterize the spline.
        :return: (Position, tangent)
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
                sum += 3 * a[i] * t ** 2 + 2 * b[i] * t + c[i]
            return np.sqrt(sum)

        while True:
            length_mid = integrate.quad(spline_derivative, 0, mid).y
            if np.abs(length_mid-length) < self.epsilon:
                break
            if length < length_mid:
                r = mid
            else:
                l = mid
            mid = (l + r) / 2

        t = mid
        return a * t**3 + b * t**2 + c * t + d, \
               3 * a * t**2 + 2 * b * t + c

    def compute_arc_length_parameterized_spline(self, num_seg):
        '''
        :param num_seg: Number of new spline segments.
        :return: list of arc length parameterized splines and segment arc lengths.
        '''
        total_length = sum(self.spline_lengths)
        seg_length = total_length / num_seg
        seg_points = []
        tangents = []

        for i in range(0, num_seg + 1):
            traversed_length = i * seg_length
            pos, tang = self._find_position_from_length(traversed_length)
            seg_points.append(pos)
            tangents.append(tang / la.norm(tang))

        arc_splines = []
        for i in range(len(seg_points) - 1):
            start_point = seg_points[i]
            end_point = seg_points[i+1]
            start_tangent = tangents[i]
            end_tangent = tangents[i+1]
            # Spline parameters
            pa = np.zeros(self.dimension)
            pb = np.zeros(self.dimension)
            pc = np.zeros(self.dimension)
            pd = np.zeros(self.dimension)
            # For each dimension, Ax=b needs to be solved to obtain parameters.
            for dim in range(self.dimension):
                pc[dim] = start_tangent[dim]
                pd[dim] = start_point[dim]
                b = np.array(
                    [end_point[dim] - start_tangent[dim] * seg_length - start_point[dim],
                     end_tangent[dim] - start_tangent[dim]])

                A = np.array([[seg_length**3, seg_length**2],
                              [3*seg_length**2, 2*seg_length]])

                ab = la.solve(A, b)
                pa[dim] = ab[0]
                pb[dim] = ab[1]

            arc_splines.append([pa, pb, pc, pd])

        return arc_splines, seg_length
