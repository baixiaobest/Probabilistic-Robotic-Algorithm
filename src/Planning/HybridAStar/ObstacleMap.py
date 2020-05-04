import numpy as np
import kdtree as kd
import src.Planning.HybridAStar.ObstacleNode as obsnode

""" Resposible for testing collision. """
class ObstacleMap:
    """
    occupancy_list: List of occupied locations, in [x, y].
    cell_size: Cell size of what each item in the occupancy list represents.
    car_geometry: Rectangular geometry of the car, parameterized by length and width. Dict {length: ..., width: ..., axel_length: }.
    path_collsion_interval: When performing collision detection of a path, the resolution of the path that will be subdivided
        into.
    k_closest: K nearest obstacles when checking collision.
    """
    def __init__(self, occupancy_list, cell_size, car_geometry, path_collsion_interval, k_closest=10):
        self.occupancy_list = occupancy_list
        self.cell_size = float(cell_size)
        self.car_geometry = car_geometry
        self.path_collsion_interval = float(path_collsion_interval)

        obstacles = [obsnode.ObstacleNode(config) for config in occupancy_list]
        self.kd_tree = kd.create(point_list=obstacles, dimensions=2)
        self.k_closest = k_closest

    def paths_are_free(self, paths):
        for path in paths:
            configs = path.generate_configs(self.path_collsion_interval)
            for config in configs:
                if self.is_occupied(config):
                    return False
        return True

    def is_occupied(self, config):
        cell_diagonal = self.cell_size * np.sqrt(2)

        # x and y is the geometric center of the car. config is on the midpoint of the rear axel of the car.
        x = config[0] + np.cos(config[2]) * self.car_geometry['axel_length'] / 2
        y = config[1] + np.sin(config[2]) * self.car_geometry['axel_length'] / 2
        theta = config[2]
        occupied = False

        results = self.kd_tree.search_knn(config[0:2], self.k_closest, ObstacleMap._kd_tree_distance)

        for kdnode, distance in results:
            # Obstacle is too far.
            if distance > self.car_geometry['length']:
                continue

            obs = kdnode.data.get_config()

            # Rotation and position of the car
            R_car = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            P_car = np.array([x, y])

            # Transform from world coordinate to car coordinate.
            T_car_world = np.identity(3)
            T_car_world[0:2, 0:2] = R_car.T
            T_car_world[0:2, 2] = -R_car.T @ P_car

            # Obstacle in world coordinate
            obs_world = np.array([obs[0], obs[1], 1])
            # Obstacle in car coordinate
            obs_car = T_car_world @ obs_world

            # X, Y distance to the car center in car coordinate
            x_d = np.abs(obs_car[0])
            y_d = np.abs(obs_car[1])

            # Axis-Aligned Bounding Box collision detection.
            if x_d < self.car_geometry['length'] / 2 + cell_diagonal \
                    and y_d < self.car_geometry['width'] / 2 + cell_diagonal:
                occupied = True
                break

        return occupied

    @staticmethod
    def _kd_tree_distance(node, end_point):
        start_point = node.get_config()
        return np.linalg.norm(np.array(start_point) - np.array(end_point))