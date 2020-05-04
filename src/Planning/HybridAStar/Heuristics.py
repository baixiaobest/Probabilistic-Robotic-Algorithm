import numpy as np
import pickle
import src.Planning.HybridAStar.DubinsCurve as dc


class DubinHeuristics:
    def __init__(self, turning_radius):
        self.turning_radius = turning_radius
        self.enable_cache = False
        self.cache_location = ""
        self.cache = None

    """
    start_config, end_config: Start and end point of the heuristics. [x, y, theta]. 
    """
    def heuristics(self, start_config, end_config):
        if self.enable_cache:
            return self._cached_heuristics(start_config, end_config)
        else:
            return self._heuristics(start_config, end_config)

    def _cached_heuristics(self, start_config, end_config):
        x_end = end_config[0]
        y_end = end_config[1]
        theta_end = end_config[2]

        R = np.array([[np.cos(theta_end), -np.sin(theta_end)],
                      [np.sin(theta_end), np.cos(theta_end)]])
        P = np.array([x_end, y_end])
        # Transform from world coordinate into end config body frame.
        T_end_world = np.identity(3)
        T_end_world[0:2, 0:2] = R.T
        T_end_world[0:2, 2] = -R.T @ P

        # Start config in end body frame.
        start_config_end_frame = T_end_world @ np.array([start_config[0], start_config[1], 1.0])
        x_start_end_frame = start_config_end_frame[0]
        y_start_end_frame = start_config_end_frame[1]
        theta_start_end_frame = (start_config[2] - end_config[2] + 2*np.pi) % (2*np.pi)

        if x_start_end_frame < self.cache['x_min'] or x_start_end_frame > self.cache['x_max']\
            or y_start_end_frame < self.cache['y_min'] or y_start_end_frame > self.cache['y_max']:
            return self._heuristics(start_config, end_config)

        row = int((x_start_end_frame - self.cache['x_min']) / self.cache['x_res'])
        col = int((y_start_end_frame - self.cache['y_min']) / self.cache['y_res'])
        height = int((theta_start_end_frame / self.cache['theta_res']))

        return self.cache['heuristics'][row, col, height]

    def _heuristics(self, start_config, end_config):
        # Dubin heuristics
        dubin = dc.DubinsCurve(start_config, end_config, self.turning_radius)
        dubin_path, dubin_length = dubin.compute()

        # Euclidean heuristics
        distance = np.linalg.norm(np.array(start_config[0:2]) - np.array(end_config[0:2]))

        return np.fmax(dubin_length, distance)

    """
    resolution: Map {x_res: , y_res: , theta_res: ,}
    limits: Map {x_min:, x_max:, y_min:, y_max:}
    save_location: Save location of generated heuristics.
    """
    def precompute_heuristics(self, resolution, limits, save_location):
        self.cache = {}
        self.cache = {**limits, **resolution}

        self.cache['x_size'] = int((limits['x_max'] - limits['x_min']) / resolution['x_res'])
        self.cache['y_size'] = int((limits['y_max'] - limits['y_min']) / resolution['y_res'])
        self.cache['theta_size'] = int(np.ceil(2 * np.pi / resolution['theta_res']))
        self.cache['heuristics'] = np.zeros((self.cache['x_size'], self.cache['y_size'], self.cache['theta_size']))

        # Precompute every scenario.
        for x in np.arange(limits['x_min'], limits['x_max'], resolution['x_res']):
            for y in np.arange(limits['y_min'], limits['y_max'], resolution['y_res']):
                for theta in np.arange(0, 2*np.pi, resolution['theta_res']):
                    row = int((x - limits['x_min']) / resolution['x_res'])
                    col = int((y - limits['y_min']) / resolution['y_res'])
                    height = int(theta / resolution['theta_res'])
                    start_config = [x, y, theta]
                    end_config = [0.0, 0.0, 0.0]
                    cost = self.heuristics(start_config, end_config)
                    self.cache['heuristics'][row, col, height] = cost

        pickle.dump(self.cache, open(save_location, "wb"))

    def load_cached_heuristics(self, file_location):
        self.cache = pickle.load(open(file_location, "rb"))
        self.enable_cache = True