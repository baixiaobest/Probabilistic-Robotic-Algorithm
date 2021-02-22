import numpy as np
from numpy.linalg import norm

""" This gets the cost function that lead to a certain coordinate x and y. """
def get_point_cost_function(target_x, target_y, x_weight, y_weight, control_weight):
    def cost_function(state, control):
        x_dist_to_target = target_x - state[0]
        y_dist_to_target = target_y - state[1]
        return x_weight * x_dist_to_target**2 + y_weight * y_dist_to_target ** 2 + control_weight * control ** 2
    return cost_function

""" 
This gets the cost function that lead to tracking of a circle trajectory. 
circle_center: loiter circle center position.
radius: loiter radius
direction: Direction of travel on circle, 1 for counter clockwise or -1 for clockwise.
direction_weight: Weight of error in direction relative to distance error
direction_tau: Direction weight exponentially decay with respect to distance.
control_weight: weight of control relative to distance error.
"""
def get_circle_cost_function(circle_center, radius, direction, direction_weight, direction_tau, control_weight):
    def cost_function(state, control):
        x = state[0]
        y = state[1]
        theta = state[2]

        dist_to_circle_center = np.sqrt((circle_center[0] - x) ** 2 + (circle_center[1] - y) ** 2)
        dist_to_circle = dist_to_circle_center - radius

        # Unit velocity
        v_unit = np.array([np.cos(theta), np.sin(theta), 0])
        # Cicle center to aircraft vector
        vec_circle_to_pos = np.array([x, y, 0]) - np.array([circle_center[0], circle_center[1], 0])
        vec_circle_to_pos_unit = vec_circle_to_pos / norm(vec_circle_to_pos)
        # Desired travel direction for aircraft.
        desired_v_vec = np.cross(np.array([0, 0, direction]), vec_circle_to_pos_unit)
        # Difference in current velocity direction and desired velocity direction, value in [0, 1].
        error_direction = norm(v_unit - desired_v_vec) / 2.0
        cost_direction_error = direction_weight * error_direction * np.exp(-direction_tau * dist_to_circle)

        return dist_to_circle ** 2 + cost_direction_error + control_weight * control ** 2
    return cost_function

"""
"""
def get_spline_cost_function(cached_spline, direction_weight, direction_tau, control_weight):
    def cost_function(state, control):
        theta = state[2]

        # Closest point on the spline.
        s, dist_to_spline = cached_spline.get_s_distance(state[0:2])

        v_unit = np.array([np.cos(theta), np.sin(theta)])
        desired_v = cached_spline.get_velocity(s)
        desired_v = desired_v / norm(desired_v)
        error_direction = norm(v_unit - desired_v) / 2.0
        direction_cost = direction_weight * error_direction * np.exp(-direction_tau * dist_to_spline)

        return dist_to_spline ** 2 + direction_cost + control_weight * control ** 2
    return cost_function

