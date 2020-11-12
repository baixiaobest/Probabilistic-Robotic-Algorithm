import numpy as np


class CachedSplineDistance:
    """
    spline: An instance of VectorCubicSpline
    configs: [{min: , max: , resolution: }...] for two dimension of the space spline is in.
    """
    def __init__(self, spline, configs):
        self.spline = spline
        self.configs = configs
        self.distance_table = None
        self.s_table = None

    def compute_cache(self):
        x_config = self.configs[0]
        y_config = self.configs[1]

        x_num_states = self._get_num_states(x_config)
        y_num_states = self._get_num_states(y_config)

        self.distance_table = np.zeros((x_num_states, y_num_states))
        self.s_table = np.zeros((x_num_states, y_num_states))

        for x_idx in range(x_num_states):
            for y_idx in range(y_num_states):
                x = x_config['min'] + x_idx * x_config['resolution']
                y = y_config['min'] + y_idx * y_config['resolution']
                s, dist = self.spline.get_s_distance(np.array([x, y]))
                self.distance_table[x_idx, y_idx] = dist
                self.s_table[x_idx, y_idx] = s

    def get_s_distance(self, point):
        if self.distance_table is None or self.s_table is None:
            self.compute_cache()

        x = point[0]
        y = point[1]
        x_config = self.configs[0]
        y_config = self.configs[1]

        if x < x_config['min'] or x > x_config['max'] or y < y_config['min'] or y > y_config['max']:
            s, dist, _ = self.spline.get_s_distance
            return s, dist
        x_idx = int((x - x_config['min']) / x_config['resolution'])
        y_idx = int((y - y_config['min']) / y_config['resolution'])

        return self.s_table[x_idx, y_idx], self.distance_table[x_idx, y_idx]

    def get_velocity(self, s):
        return self.spline.get_velocity(s)

    def _get_num_states(self, config):
        num_states = int((config['max'] - config['min']) / config['resolution']) + 1
        return num_states
