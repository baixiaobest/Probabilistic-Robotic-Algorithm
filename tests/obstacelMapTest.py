import numpy as np
import unittest
import src.Planning.HybridAStar.ObstacleMap as obmap

def get_obstacle_map():
    obstacles = [np.array([5.0, 4.0]), np.array([6.0, 4.0]), np.array([7.0, 4.0])]
    car_geometry = {'length': 2.0, 'width': 1.0, 'axel_length': 1.0}
    path_resolution = 0.1
    cell_size = 1.0
    return obmap.ObstacleMap(obstacles, cell_size, car_geometry, path_resolution)

class ObstacleMapTest(unittest.TestCase):
    def test_occupied_1(self):
        obstacle_map = get_obstacle_map()
        config = [5.0, 4.0, np.pi / 2]
        self.assertEqual(True, obstacle_map.is_occupied(config))

    def test_occupied_2(self):
        obstacle_map = get_obstacle_map()
        config = [5.0, 3.0, np.pi / 2]
        self.assertEqual(True, obstacle_map.is_occupied(config))

    def test_free_1(self):
        obstacle_map = get_obstacle_map()
        config = [5.0, 2.5 - np.sqrt(2), np.pi / 2]
        self.assertEqual(False, obstacle_map.is_occupied(config))

    def test_free_2(self):
        obstacle_map = get_obstacle_map()
        config = [5.0, 4 - np.sqrt(2) - 0.501, 0]
        self.assertEqual(False, obstacle_map.is_occupied(config))

    def test_free_3(self):
        obstacle_map = get_obstacle_map()
        config = [2.0, 4, np.pi / 2]
        self.assertEqual(False, obstacle_map.is_occupied(config))

    def test_occupied_3(self):
        obstacle_map = get_obstacle_map()
        config = [5.0, 4 - np.sqrt(2) - 0.501, 0.1]
        self.assertEqual(True, obstacle_map.is_occupied(config))

if __name__=="__main__":
    unittest.main()