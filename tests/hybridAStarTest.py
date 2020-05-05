import numpy as np
from matplotlib import pyplot as plt
import src.Utils.plot as uplt
import src.Planning.HybridAStar.FowardNonHolonomicMotionModel as model
import src.Planning.HybridAStar.ObstacleMap as obmap
import src.Planning.HybridAStar.Heuristics as h
import src.Planning.HybridAStar.HybridAStar as hastar
import cProfile, pstats

grid_resolution = 0.5

def show(configs, paths, x_low, x_high, y_low, y_high, point_interval=0.2):
    for path in paths:
        points = path.generate_points(point_interval)
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.plot(x, y)
    uplt.limit(x_low, x_high, y_low, y_high)
    uplt.plotRobotPoses(configs)

def get_motion_model():
    wheel_max_angle = 20.0 * np.pi / 180.0
    num_angle_controls = 5
    car_axel_length = 1.0
    distance_increment = 2
    motion_model = model.FowardNonHolonomicMotionModel(wheel_max_angle,
                                                       num_angle_controls,
                                                       car_axel_length,
                                                       distance_increment)
    return motion_model

def get_obstacle_map():
    car_geometry = {'length': 2.0, 'width': 1.0, 'axel_length': 1.0}
    path_collsion_interval = 2.0
    cell_size = grid_resolution

    obstacles = []
    map = uplt.readBMPAsNumpyArray("../map/small_obstacle_map.bmp")
    for row in range(len(map)):
        for col in range(len(map[0])):
            if map[row, col] == 0:
                x = cell_size * col
                y = cell_size * row
                obstacles.append(np.array([x, y]))

    return obmap.ObstacleMap(obstacles, cell_size, car_geometry, path_collsion_interval), map

def get_heuristics():
    heuristics = h.DubinHeuristics(motion_model.get_min_turning_radius())
    h_resolution = {'x_res': 1.0, 'y_res': 1.0, 'theta_res': np.pi / 4.0}
    h_limits = {'x_min': -50.0, 'x_max': 50.0, 'y_min': -50.0, 'y_max': 50.0}
    # heuristics.precompute_heuristics(h_resolution, h_limits, "../cache/heuristics")
    heuristics.load_cached_heuristics("../cache/heuristics")
    return heuristics

if __name__=="__main__":
    pr = cProfile.Profile()
    start_config = np.array([0.0, 0.0, 0.0])
    end_config = np.array([48.0, 22.0, -np.pi/2])

    motion_model = get_motion_model()
    obstacle_map, map = get_obstacle_map()

    heuristics = get_heuristics()

    resolutions = {'x_res': 2.0, 'y_res': 2.0, 'theta_res': np.pi / 2}
    limits = {'x_min': 0, 'x_max': 20.0, 'y_min': 0, 'y_max': 30.0}
    planner = hastar.HybridAStar(obstacle_map, heuristics, motion_model, resolutions, limits,
                                 max_num_nodes=20000, bucket_interval=5.0)

    print("start")
    pr.enable()
    planner.compute(start_config, end_config)
    pr.disable()
    path = planner.get_path()
    all_path = planner.get_all_explored_path()

    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats()

    uplt.plotOccupancyGrid(map, grid_resolution)
    show([start_config, end_config], all_path, -30, 50, -30, 50)
    uplt.show()
    uplt.plotOccupancyGrid(map, grid_resolution)
    show([start_config, end_config], path, -30, 50, -30, 50, point_interval=0.1)
    uplt.show()
