import numpy as np

""" This gets the cost function that lead to a certain coordinate x and y. """
def get_point_cost_function(target_x, target_y, x_weight, y_weight, control_weight):
    def cost_function(state, control):
        x_dist_to_target = target_x - state[0]
        y_dist_to_target = target_y - state[1]
        return x_weight * x_dist_to_target**2 + y_weight * y_dist_to_target ** 2 + control_weight * control ** 2
    return cost_function

""" This gets the cost function that lead to tracking of a circle trajectory. """
def get_circle_cost_function(circle_center, radius, control_weight):
    def cost_function(state, control):
        x = state[0]
        y = state[1]
        dist_to_circle_center = np.sqrt((circle_center[0] - x) ** 2 + (circle_center[1] - y) ** 2)
        return (dist_to_circle_center - radius) ** 2 + control_weight * control ** 2
    return cost_function