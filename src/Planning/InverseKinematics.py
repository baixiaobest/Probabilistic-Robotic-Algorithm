import numpy as np
from scipy.optimize import minimize
import src.Planning.PotentialField as pf
from scipy import optimize
import matplotlib as plt

'''
Inverse kinematics for two links robotic arm with 2 degree of freedom.
'''
class InverseKinematics:
    '''
    L1, L2: length of two links, in meter.
    limit1, limit2: joint limit of two joints, tuple of (min, max).
    origin: Workspace position of the base of the robotic arm.
    '''
    def __init__(self, L1, L2, limit1, limit2, origin):
        self.L1 = L1
        self.L2 = L2
        self.limit1 = limit1
        self.limit2 = limit2
        self.origin = origin
        self.goal = None

        self.obstacle_map = None
        self.resolution = None
        self.potential_field = None
        self.num_control_points = 1

    def _forward_kinematics(self, theta1, theta2):
        return np.array([self.L1 * np.cos(theta1) + self.L2 * np.cos(theta1 + theta2),
                         self.L1 * np.sin(theta1) + self.L2 * np.sin(theta1 + theta2)])\
                + self.origin

    ''' Quadratic objective function. '''
    def _obstacle_free_objective_function(self, config):
        curr_xy = self._forward_kinematics(config[0], config[1])
        return np.linalg.norm(curr_xy - self.goal) ** 2

    def _get_control_points(self, theta1, theta2):
        pt1 = self.origin
        pt2 = self.origin + np.array([self.L1 * np.cos(theta1), self.L1 * np.sin(theta1)])
        pt3 = pt2 + np.array([self.L2 * np.cos(theta1 + theta2), self.L2 * np.sin(theta1 + theta2)])
        pts = [pt1, pt2, pt3]

        control_points = []

        for i in range(len(pts) - 1):
            start = pts[i]
            end = pts[i + 1]
            direction = end - start
            # Split each robotic arm into num_control_points + 1 segments.
            for j in range(self.num_control_points):
                control_points.append(direction * (j + 1) / (self.num_control_points + 1.0) + start)

        return control_points

    '''
    Objective function is now the potential field.
    '''
    def _obstacle_objective_function(self, config):
        theta1, theta2 = config
        control_points = self._get_control_points(theta2, theta2)
        objective_value = 0
        for point in control_points:
            col, row = int(point[0] / self.resolution), int(point[1] / self.resolution)
            objective_value = objective_value + self.potential_field[row, col]
        return objective_value

    '''
    Given x, y position of end effector, calculate configuration.
    return numpy array of solution.
    '''
    def calculate_obstacle_free_ik(self, x, y):
        self.goal = np.array([x, y])
        options = {'gtol': 1e-9}
        res = minimize(self._obstacle_free_objective_function, np.array([0.0, 0.0]), method='SLSQP',
                       bounds=[self.limit1, self.limit2], tol=1e-4, options=options)

        if not res.success:
            raise Warning("Solution to IK is not found")

        return res.x

    '''
    Given x, y position of end effector, calculate configuration.
    return numpy array of solution. This function considers obstacle.
    obstacle_map: obstacle_map in workspace.
    resolution: resolution of the obstacle_map, meter/pixel.
    num_control_points: number of control points in each arm segment for objective
        function calculation.
    '''
    def calculate_obstacle_ik(self, x, y, obstacle_map, resolution, num_control_points):
        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.goal = np.array([x, y])
        self.num_control_points = num_control_points

        params = {"zeta": 1, "d_goal": 10, "eta": 5, "q_star": 5, "max_potential": 200, "max_gradient": 10}
        potential_field_gen = pf.PotentialField(obstacle_map, params, resolution, connectivity='vertex')
        potential_field_gen.set_goal(self.goal)
        self.potential_field = potential_field_gen.get_potential_field()

        res = optimize.shgo(self._obstacle_objective_function, bounds=[self.limit1, self.limit2])

        return res.x
