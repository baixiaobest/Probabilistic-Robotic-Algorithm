import numpy as np
from matplotlib import pyplot as plt
import src.Utils.plot as uplt
import src.Planning.HybridAStar.FowardNonHolonomicMotionModel as model

point_interval = 0.1

def show(configs, paths, x_low, x_high, y_low, y_high):
    for path in paths:
        points = path[0].generate_points(point_interval)
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.plot(x, y)
    uplt.limit(x_low, x_high, y_low, y_high)
    uplt.plotRobotPoses(configs)
    plt.show()

if __name__=="__main__":
    wheel_max_angle = 20.0 * np.pi/180.0
    num_angle_controls = 10
    car_axel_length = 1.0
    distance_increment = 10.0
    motion_model = model.FowardNonHolonomicMotionModel(wheel_max_angle,
                                                       num_angle_controls,
                                                       car_axel_length,
                                                       distance_increment)

    start_config = [5.0, 5.0, 0]
    configs, paths, paths_length = motion_model.generate_neighbors(start_config)
    configs.append(start_config)
    show(configs, paths, 0, 25, -10, 15)

