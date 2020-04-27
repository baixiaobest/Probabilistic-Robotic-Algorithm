import numpy as np
import src.Utils.plot as plot
import matplotlib.pyplot as plt
from matplotlib import collections as c
import src.Planning.PotentialField.InverseKinematics as ik
from matplotlib import colors as mcolors

if __name__ == "__main__":
    obstacle_map = plot.readBMPAsNumpyArray("../map/ik_obstacle_map.bmp")
    h, w = obstacle_map.shape
    resolution = 0.05 * 6.0
    L1 = 10
    L2 = 5
    origin = np.array([15, 15])

    ik_cal = ik.InverseKinematics(L1=L1, L2=L2, limit1=(-2*np.pi, 2*np.pi),
                                  limit2=(-2*np.pi, 2*np.pi), origin=origin)

    # goal = np.array([0, 15]) + origin
    # theta1, theta2 = ik_cal.calculate_obstacle_ik(x=goal[0], y=goal[1], obstacle_map=obstacle_map,
    #                                               resolution=resolution, num_control_points=1)
    # plt.imshow(ik_cal.potential_field)
    # plt.show()

    goal = np.array([1, -6]) + origin
    theta1, theta2 = ik_cal.calculate_obstacle_free_ik(x=goal[0], y=goal[1])

    print("{0} {1}".format(theta1, theta2))

    pt1 = np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)]) + origin
    pt2 = np.array([L2 * np.cos(theta1 + theta2), L2 * np.sin(theta1 + theta2)]) + pt1
    lines = [[origin, pt1], [pt1, pt2]]

    fig, ax = plt.subplots()
    plot.plotOccupancyGrid(obstacle_map, resolution)
    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    lc = c.LineCollection(lines, colors=colors)
    ax.add_collection(lc)

    plt.show()