import numpy as np
import src.Planning.HybridAStar.Path as p
import src.Planning.HybridAStar.DubinCircle as dc


class FowardNonHolonomicMotionModel:
    """
    wheel_max_angle: Maximum turning angle of the car.
    num_angle_controls: Number of discrete wheel controls.
    car_axis_length: Distance from front axel to rear axel.
    distance_increment: distance traversed after applying action.
        This distance should be less than the circumference of the smallest turning circle.
    """
    def __init__(self, wheel_max_angle, num_angle_controls, car_axel_length, distance_increment):
        self.wheel_max_angle = float(wheel_max_angle)
        self.num_angle_controls = int(num_angle_controls)
        self.car_axel_length = float(car_axel_length)
        self.distance_increment = float(distance_increment)

        wheel_increment = 2 * wheel_max_angle / (self.num_angle_controls - 1)
        self.wheel_actions = [theta for theta in np.arange(-wheel_max_angle, wheel_max_angle + 0.001, wheel_increment)]

    """
    config: Configuration of the car, [x, y, theta]. Center of the car is located at
        the midpoint of the rear axel of the car.
    """
    def generate_neighbors(self, config):
        new_configs = []
        paths_length = []
        paths = []
        x_axis = np.array([1.0, 0.0])

        for wheel_theta in self.wheel_actions:
            # positive if turning left, zero if not turning, negative if turning right.
            turn_dir = np.sign(wheel_theta)

            if np.abs(wheel_theta) > 0.0001:
                turning_radius = np.abs(self.car_axel_length / np.tan(wheel_theta))
                # Angle at which the car turns after this action.
                angle_turned = turn_dir * self.distance_increment / turning_radius

                # Vector from car rear axel position to center of rotation.
                vec_car_CoR = np.array([np.cos(config[2] + turn_dir * np.pi / 2),
                                        np.sin(config[2] + turn_dir * np.pi / 2)]) * turning_radius
                vec_CoR_car = -vec_car_CoR
                # Center of rotation.
                CoR = config[0:2] +  vec_car_CoR

                vec_CoR_new_car = np.array([[np.cos(angle_turned), -np.sin(angle_turned)],
                                            [np.sin(angle_turned), np.cos(angle_turned)]]) \
                                  @ vec_CoR_car
                # New center of the car
                new_xy = CoR + vec_CoR_new_car

                new_theta = (config[2] + angle_turned) % (2 * np.pi)
                new_configs.append(np.array([new_xy[0], new_xy[1], new_theta]))

                # Generate path.
                path_start_theta = p.angle_diff(vec_CoR_car, x_axis)
                path_end_theta = p.angle_diff(vec_CoR_new_car, x_axis)
                circle_type = dc.CircleType.COUNTER_CLOCKWISE if turn_dir >= 0 else dc.CircleType.CLOCKWISE
                paths.append([p.CircularPath(CoR, path_start_theta, path_end_theta, circle_type, turning_radius)])
            else:
                vec_car_front = np.array([np.cos(config[2]), np.sin(config[2])])
                new_xy = config[0:2] + self.distance_increment * vec_car_front
                new_configs.append(np.array([new_xy[0], new_xy[1], config[2]]))

                paths.append([p.StraightPath(config[0:2], new_xy)])

            paths_length.append(self.distance_increment)

        return new_configs, paths, paths_length


    def get_min_turning_radius(self):
        return np.abs(self.car_axel_length / np.tan(self.wheel_max_angle))