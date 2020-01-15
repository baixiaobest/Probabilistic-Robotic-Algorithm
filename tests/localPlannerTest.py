import numpy as np
import src.Utils.plot as plot
import matplotlib.pyplot as plt
import src.Planning.RectangleLocalPlanner as rlp
import src.Planning.RandomConfigSampler as rcs

if __name__ == '__main__':
    obstacle_map = plot.readBMPAsNumpyArray("../map/small_obstacle_map.bmp")
    resolution = 0.05

    step_size = (0.2, 0.1)
    robot_width = 0.3
    robot_length = 0.6
    local_planner = rlp.RectangleLocalPlanner(obstacle_map, resolution, step_size, robot_width, robot_length)

    config_sampler = rcs.RandomConfigSampler(local_planner)
    num_samples = 20

    # Uniformly sample configurations.

    uniform_configs = []
    for i in range(num_samples):
        uniform_configs.append(config_sampler.uniform_collision_free_sample())

    plot.plotOccupancyGrid(obstacle_map, resolution)
    plot.plotRectangles(uniform_configs, robot_width, robot_length)
    plot.show()

    # Sample around obstacles.
    sigmas = (0.1, 0.1, 0.5)
    biased_configs = []
    num_biased_samples = 20
    for i in range(num_biased_samples):
        new_config = config_sampler.sample_around_obstacle(sigmas, trials=200)
        if new_config is not None:
            biased_configs.append(new_config)

    plot.plotOccupancyGrid(obstacle_map, resolution)
    plot.plotRectangles(biased_configs, robot_width, robot_length)
    plot.show()

    # Check all possible connection.
    all_configs = uniform_configs + biased_configs
    connections = []
    for i in range(len(all_configs)):
        for j in range(i, len(all_configs)):
            if local_planner.check_connection(all_configs[i], all_configs[j]):
                connections.append([all_configs[i][0:2], all_configs[j][0:2]])

    plot.plotOccupancyGrid(obstacle_map, resolution)
    plot.plotRectangles(all_configs, robot_width, robot_length)
    plot.plotLines(connections)
    plot.show()
