import numpy as np
import src.Utils.plot as plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import src.Planning.PotentialField as pf

if __name__ == '__main__':
    obstacle_map = plot.readBMPAsNumpyArray("../map/small_obstacle_map.bmp")
    h, w = obstacle_map.shape
    resolution = 0.05 * 6.0
    params = {"zeta": 1, "d_goal": 5, "eta": 5, "q_star": 5, "max_potential": 200, "max_gradient": 10}
    goal = (20, 20)

    potential_field_gen = pf.PotentialField(obstacle_map, params, topology='euclidean', connectivity='vertex', resolution=resolution)
    potential_field_gen.set_goal(goal)

    obstacle_dist_map = potential_field_gen.get_obstacle_distance_map()
    repulsive_potential_field = potential_field_gen.get_repulsive_potential_field()
    attractive_potential_field = potential_field_gen.get_attractive_potential_field()
    total_potential_field = potential_field_gen.get_potential_field()

    plt.figure()
    plt.imshow(obstacle_dist_map, cmap='jet')
    plt.figure()
    plt.imshow(repulsive_potential_field, cmap='jet')

    plt.figure()
    plt.imshow(attractive_potential_field, cmap='jet')

    plt.figure()
    plt.imshow(total_potential_field, cmap='jet')

    potential_field_gen.get_negative_gradient((20, 21))
    potential_field_gen.get_negative_gradient((44 * resolution, 34 * resolution))
    gradient_x, gradient_y = potential_field_gen.get_negative_gradient_field()

    X = np.arange(0, w, 1)
    Y = np.arange(0, h, 1)
    plt.figure()
    plt.quiver(X, Y, gradient_x, gradient_y)

    plt.show()