import numpy as np


"""
A local planner for a rectangular shape robot.
"""
class RectangleLocalPlanner:
    '''
    obstacle_map: 2d np array of the world containing obstacles.
    resolution: Resolution of the map in meter/cell.
    step_size: Step size used when checking connection between two config.
        Tuple of (xy_step_size (meter), theta_step_size (radian)).
    width: Width of the rectangular shape robot, in meter.
    length: Length of the rectangular shape robot, in meter.
    '''
    def __init__(self, obstacle_map, resolution, step_size, width, length):
        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.step_size = step_size
        self.width = width
        self.length = length
        self.OBSTACLE_NUM = 0

    '''
    Check if a given config (x, y, theta) of the robot is in collision with the obstacle.
    Return true if it is in collision.
    '''
    def is_in_collision(self, config):
        # Five control points for collision detection.
        center = np.array([0, 0])
        front = np.array([0.5 * self.length, 0])
        back = np.array([-0.5 * self.length, 0])
        left = np.array([0, 0.5 * self.width])
        right = np.array([0, -0.5 * self.width])
        front_left = np.array([0.5 * self.length, 0.5 * self.width])
        front_right = np.array([0.5 * self.length, -0.5 * self.width])
        rear_left = np.array([-0.5 * self.length, 0.5 * self.width])
        rear_right = np.array([-0.5 * self.length, -0.5 * self.width])

        # Assume obstacles are always larger than the robot.
        # So we can use 9 control points to check collision.
        control_points = [center, front, back, left, right, front_left, front_right, rear_left, rear_right]

        for control_point in control_points:
            # Calculate the position of the control points given a configuration.
            x, y, theta = config
            pos = np.array([
                control_point[0] * np.cos(theta) - control_point[1] * np.sin(theta) + x,
                control_point[0] * np.sin(theta) + control_point[1] * np.cos(theta) + y])

            col, row = int(pos[0] / self.resolution), int(pos[1] / self.resolution)

            height, width = self.obstacle_map.shape

            # When control point is outside of the obstacle map, consider it as
            # in collision.
            if row >= height or row < 0 or col >= width or col < 0 \
                    or self.obstacle_map[row, col] == self.OBSTACLE_NUM:
                return True

        return False

    '''
    Check the connection from config_start to config_end is collision free.
    Return true if connected.
    config_start, config_end: Tuple of (x, y, theta).
    '''
    def check_connection(self, config_start, config_end):
        # Both configurations need to be free.
        if self.is_in_collision(config_start) or self.is_in_collision(config_end):
            return False

        xy_step_size, theta_step_size = self.step_size
        connected = False
        current_config = config_start
        in_collision = self.is_in_collision(current_config)

        while not in_collision:
            # Diff in xy values from current config to end config.
            xy_diff = np.array([config_end[0] - current_config[0], config_end[1] - current_config[1]])

            # If distance from current to config_end is less than step size,
            # then we complete all the checks and the two configurations are connected.
            if np.linalg.norm(xy_diff) < xy_step_size:
                connected = True
                break

            # Step forward the configuration by xy_step_size.
            xy_norm = xy_diff / np.linalg.norm(xy_diff)
            new_xy = xy_norm * xy_step_size + np.array([current_config[0], current_config[1]])
            current_config = (new_xy[0], new_xy[1], current_config[2])

            # Try to find a theta configuration that the robot is not in collision.
            total_theta = 0
            in_collision = self.is_in_collision(current_config)
            while in_collision and total_theta <= 2 * np.pi:
                current_config = (current_config[0], current_config[1], current_config[2] + theta_step_size)
                in_collision = self.is_in_collision(current_config)
                total_theta = total_theta + theta_step_size

        return connected

    def get_xy_limit(self):
        height, width = self.obstacle_map.shape
        return (0, width * self.resolution), (0, height * self.resolution)
