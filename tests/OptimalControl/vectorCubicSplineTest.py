import unittest
import src.OptimalControl.VectorCubicSpline as cubic
import numpy as np
from numpy.linalg import norm


class MyTestCase(unittest.TestCase):

    def spline_test(self, point, a0, a1, a2, a3):
        spline = cubic.VectorCubicSpline(a0, a1, a2, a3)

        points_on_spline = []
        for s in np.linspace(0, 1, 30):
            points_on_spline.append(spline.get_point(s))
        # Calculate minimum distance and point on the spline.
        s, min_dist = spline.get_s_distance(point)
        min_point = spline.get_point(s)

        min_dist_recalculated = norm(point - min_point)

        self.assertAlmostEqual(min_dist, min_dist_recalculated)

        for point_on_spline in points_on_spline:
            dist_to_spline = norm(point - point_on_spline)
            self.assertTrue(dist_to_spline >= min_dist)

    def test_spline_0(self):
        self.spline_test(np.array([10, 10]), [3.5, 2.7], [2.3, 5], [-4.3, -2], [2.5, 3.8])

    def test_spline_1(self):
        self.spline_test(np.array([0, 0]), [3.5, 2.7], [2.3, 5], [-4.3, -2], [2.5, 3.8])

    def test_spline_2(self):
        self.spline_test(np.array([5, 5]), [3.5, 2.7], [2.3, 5], [-4.3, -2], [2.5, 3.8])

if __name__ == '__main__':
    unittest.main()
