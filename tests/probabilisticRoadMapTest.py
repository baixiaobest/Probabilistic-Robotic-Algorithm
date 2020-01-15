import numpy as np
import src.Utils.plot as plot
import matplotlib.pyplot as plt
import src.Planning.RectangleLocalPlanner as rlp
import src.Planning.ProbabilisticRoadmap as prm
import random

if __name__ == '__main__':
    random.seed(0)
    obstacle_map = plot.readBMPAsNumpyArray("../map/small_obstacle_map.bmp")
    resolution = 0.1

    step_size = (0.2, 0.1)
    robot_width = 0.3
    robot_length = 0.6
    local_planner = rlp.RectangleLocalPlanner(obstacle_map, resolution, step_size, robot_width, robot_length)

    k_closest = 8
    distance = 5
    num_nodes = 100
    roadmap = prm.ProbabilisticRoadmap(local_planner, k_closest, distance, num_nodes)
    roadmap.compute_roadmap()

    configs = roadmap.get_all_configs()
    connections = roadmap.get_all_connections()
    # Ignore theta value in the connections.
    connections = [[conn[0][0:2], conn[1][0:2]] for conn in connections]

    plot.plotOccupancyGrid(obstacle_map, resolution)
    plot.plotRectangles(configs, robot_width, robot_length)
    plot.plotLines(connections)
    plot.show()
    start_config = (9, 8, 0.5)
    goal_config = (7, 1, 1)
    path = roadmap.query(start_config, goal_config)
    path_lines = []
    for i in range(len(path) - 1):
        path_lines.append([path[i][0:2], path[i+1][0:2]])

    plot.plotOccupancyGrid(obstacle_map, resolution)
    plot.plotRectangles(configs, robot_width, robot_length)
    plot.plotLines(path_lines, (0.1, 1, 1, 1))
    plot.show()