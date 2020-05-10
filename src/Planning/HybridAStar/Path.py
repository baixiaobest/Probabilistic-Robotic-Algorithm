import numpy as np
import src.Planning.HybridAStar.DubinCircle as dc

class CircularPath:
    def __init__(self, position, start_theta, end_theta, circle_type, radius):
        self.position = position
        self.start_theta = (start_theta + 2 * np.pi) % (2 * np.pi)
        self.end_theta = (end_theta + 2 * np.pi) % (2 * np.pi)
        self.cirle_type = circle_type
        self.radius = float(radius)

    def length(self):
        return self._path_theta() * self.radius

    def generate_points(self, point_interval):
        configs = self.generate_configs(point_interval)
        return [config[0:2] for config in configs]

    def generate_configs(self, point_interval):
        theta_invteral = point_interval / self.radius
        path_theta = self._path_theta()
        num_points = int(path_theta / theta_invteral) + 1

        x_axis = np.array([1.0, 0])
        configs = []
        for i in range(num_points):
            theta = 0
            if self.cirle_type == dc.CircleType.COUNTER_CLOCKWISE:
                theta = self.start_theta + i * theta_invteral
            else:
                theta = self.start_theta - i * theta_invteral

            point = self.position \
                    + np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]]) @ x_axis * self.radius
            config = np.array([point[0], point[1], theta])
            configs.append(config)
        return configs

    """ The angular distance from starting position to end position. """
    def _path_theta(self):
        path_theta = 0
        if self.cirle_type == dc.CircleType.COUNTER_CLOCKWISE:
            theta_end_start = self.end_theta - self.start_theta
            if theta_end_start >= 0:
                path_theta = theta_end_start
            else:
                path_theta = 2 * np.pi + theta_end_start
        # Clockwise path
        else:
            theta_start_end = self.start_theta - self.end_theta
            if theta_start_end >= 0:
                path_theta = theta_start_end
            else:
                path_theta = 2 * np.pi + theta_start_end
        return path_theta


class StraightPath:
    def __init__(self, start_position, end_position):
        self.start_position = start_position
        self.end_position = end_position

    def length(self):
        return np.linalg.norm(self.end_position - self.start_position)

    def generate_points(self, point_interval):
        configs = self.generate_configs(point_interval)
        return [config[0:2] for config in configs]

    def generate_configs(self, point_interval):
        vec_start_end = self.end_position - self.start_position
        distance = np.linalg.norm(vec_start_end)
        vec_start_end_normalized = vec_start_end / distance
        num_points = int(distance / point_interval) + 1
        theta = angle_diff(vec_start_end_normalized, np.array([1.0, 0]))

        configs = []

        for i in range(num_points):
            new_point = self.start_position + i * point_interval * vec_start_end_normalized
            config = np.array([new_point[0], new_point[1], theta])
            configs.append(config)

        return configs

""" Angle difference between two vector. Return signed angle difference. """
def angle_diff(a, b):
    theta = np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    cross = np.cross(a, b)
    if cross > 0:
        theta = -theta
    return theta