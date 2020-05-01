import numpy as np
import src.Planning.HybridAStar.DubinsCurve as dubin
from matplotlib import pyplot as plt
import src.Utils.plot as uplt
import unittest
import random

def generate_cw_csc_test_1():
    turning_radius = 5
    sqrt_2 = np.sqrt(2)
    start = np.array([-5 / sqrt_2, -5 / sqrt_2, to_rad(135)])
    end = np.array([30 + 5 / sqrt_2, -5 / sqrt_2, to_rad(225)])
    dc = dubin.DubinsCurve(start, end, turning_radius)
    return dc

def generate_cw_csc_test_2():
    turning_radius = 5
    sqrt_2 = np.sqrt(2)
    start = np.array([0, -5, to_rad(180)])
    end = np.array([30 / sqrt_2 + 5, 30 / sqrt_2, to_rad(270)])
    dc = dubin.DubinsCurve(start, end, turning_radius)
    return dc

def generate_ccw_csc_test_1():
    turning_radius = 5
    sqrt_2 = np.sqrt(2)
    start = np.array([-5/sqrt_2, 5/sqrt_2, to_rad(225)])
    end = np.array([30 + 5/sqrt_2, -5/sqrt_2, to_rad(45)])
    dc = dubin.DubinsCurve(start, end, turning_radius)
    return dc

def generate_cw_ccw_csc_test():
    turning_radius = 5
    sqrt_2 = np.sqrt(2)
    start = np.array([-5 / sqrt_2, -5 / sqrt_2, to_rad(135)])
    end = np.array([30 + 5 / sqrt_2, -5 / sqrt_2, to_rad(45)])
    dc = dubin.DubinsCurve(start, end, turning_radius)
    return dc

def generate_ccw_cw_csc_test():
    turning_radius = 5
    sqrt_2 = np.sqrt(2)
    start = np.array([-5 / sqrt_2, 5 / sqrt_2, to_rad(225)])
    end = np.array([30 + 5 / sqrt_2, -5 / sqrt_2, to_rad(225)])
    dc = dubin.DubinsCurve(start, end, turning_radius)
    return dc

def generate_ccw_csc_test_2():
    turning_radius = 5
    sqrt_2 = np.sqrt(2)
    start = np.array([0, 5, to_rad(180)])
    end = np.array([30/sqrt_2, -30/sqrt_2 - 5, to_rad(0)])
    dc = dubin.DubinsCurve(start, end, turning_radius)
    return dc

def generate_ccc_test_1():
    turning_radius = 5
    sqrt_3 = np.sqrt(3)
    start = np.array([5/2, -5/2 * sqrt_3, to_rad(30)])
    end = np.array([5,  0, to_rad(270)])
    dc = dubin.DubinsCurve(start, end, turning_radius)
    return dc

def generate_ccc_test_2():
    turning_radius = 5
    sqrt_3 = np.sqrt(3)
    start = np.array([5/2 * sqrt_3, 5 / 2, to_rad(120)])
    end = np.array([5, 0, to_rad(270)])
    dc = dubin.DubinsCurve(start, end, turning_radius)
    return dc

def generate_random_test():
    turning_radius = 5
    x_lower = 2*turning_radius
    x_high = 50 - 2*turning_radius
    y_lower = 2*turning_radius
    y_high = 50 - 2*turning_radius
    start = np.array([random.uniform(x_lower, x_high),
                      random.uniform(y_lower, y_high),
                      random.uniform(0, 2 * np.pi)])
    end = np.array([random.uniform(x_lower, x_high),
                      random.uniform(y_lower, y_high),
                      random.uniform(0, 2 * np.pi)])
    dc = dubin.DubinsCurve(start, end, turning_radius)
    return dc

def show(dubincurve, x_low, x_high, y_low, y_high, no_plot=False):
    dubincurve.compute()
    points = dubincurve.generate_points()
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.plot(x, y)
    uplt.limit(x_low, x_high, y_low, y_high)
    uplt.plotRobotPoses([dubincurve.start, dubincurve.end])
    if not no_plot:
        plt.show()

def compare_length(length_1, length_2, tolerance=0.01):
    return np.abs(length_1 - length_2) < tolerance

def to_rad(deg):
    return deg * np.pi / 180.0

class DubinCurveTest(unittest.TestCase):
    def test_cw_csc_1(self):
        dc = generate_cw_csc_test_1()
        path, length = dc.compute()
        correct_length = 53.562
        self.assertEqual(True, compare_length(correct_length, length), "cw_csc_test_1 failed")

    def test_cw_csc_2(self):
        dc = generate_cw_csc_test_2()
        path, length = dc.compute()
        correct_length = 53.562
        self.assertEqual(True, compare_length(correct_length, length), "cw_csc_test_2 failed")

    def test_ccw_csc_1(self):
        dc = generate_ccw_csc_test_1()
        path, length = dc.compute()
        correct_length = 5 * np.pi + 30
        self.assertEqual(True, compare_length(correct_length, length), "ccw_csc_test_1 failed")

    def test_ccw_csc_2(self):
        dc = generate_ccw_csc_test_2()
        path, length = dc.compute()
        correct_length = 5 * np.pi + 30
        self.assertEqual(True, compare_length(correct_length, length), "ccw_csc_test_2 failed")

    def test_cw_ccw_csc(self):
        dc = generate_cw_ccw_csc_test()
        path, length = dc.compute()
        theta = np.arccos(5/15)
        correct_length = 5 * np.pi + 2 * 15 * np.sin(theta) + 2 * 5 * (np.pi / 2 - theta)
        self.assertEqual(True, compare_length(correct_length, length), "cw_ccw_csc_test failed")

    def test_ccw_cw_csc(self):
        dc = generate_ccw_cw_csc_test()
        path, length = dc.compute()
        theta = np.arccos(5/15)
        correct_length = 3 / 4 * 10 * np.pi + 2 * 15 * np.sin(theta) + 2 * 5 * (np.pi / 2 - theta)
        self.assertEqual(True, compare_length(correct_length, length), "ccw_cw_csc_test failed")

    def test_ccc_1(self):
        dc = generate_ccc_test_1()
        path, length = dc.compute()
        correct_length = 2 * 5 * np.pi
        self.assertEqual(True, compare_length(correct_length, length), "cccc_test_1 failed")

    def test_ccc_2(self):
        dc = generate_ccc_test_2()
        path, length = dc.compute()
        correct_length = (1 + 30/360) * 2 * 5 * np.pi
        self.assertEqual(True, compare_length(correct_length, length), "cccc_test_2 failed")

if __name__=="__main__":
    show(generate_cw_csc_test_1(), -10, 40, -25, 25)
    show(generate_cw_csc_test_2(), -10, 30, -10, 30)
    show(generate_ccw_csc_test_1(), -10, 40, -25, 25)
    show(generate_ccw_csc_test_2(), -10, 30, -30, 10)
    show(generate_cw_ccw_csc_test(), -10, 40, -25, 25)
    show(generate_ccw_cw_csc_test(), -10, 40, -25, 25)
    show(generate_ccc_test_1(), -0, 20, -10, 10)
    show(generate_ccc_test_2(), -5, 15, -5, 15)
    for i in range(10):
        show(generate_random_test(), 0, 50, 0, 50, no_plot=True)
    plt.show()
    unittest.main()