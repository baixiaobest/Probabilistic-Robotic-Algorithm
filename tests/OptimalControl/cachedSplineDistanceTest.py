import unittest
import src.OptimalControl.VectorCubicSpline as cs
import src.OptimalControl.CachedSplineDistance as ccs
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        spline = cs.VectorCubicSpline([3.5, 2.7], [2.3, 5], [-4.3, -2], [2.5, 3.8])
        configs = [{'min': 0, 'max': 20, 'resolution': 0.5}, {'min': 0, 'max': 20, 'resolution': 0.5}]
        cached_spline = ccs.CachedSplineDistance(spline, configs)
        cached_spline.compute_cache()

        for x_idx in range(int((configs[0]['max'] - configs[0]['min']) / configs[0]['resolution'])):
            for y_idx in range(int((configs[1]['max'] - configs[1]['min']) / configs[1]['resolution'])):
                x = configs[0]['min'] + x_idx * configs[0]['resolution']
                y = configs[1]['min'] + y_idx * configs[1]['resolution']
                point = np.array([x, y])
                _, spline_dist = spline.get_s_distance(point)
                _, cached_dist = cached_spline.get_s_distance(point)

                self.assertAlmostEqual(spline_dist, cached_dist)


if __name__ == '__main__':
    unittest.main()
