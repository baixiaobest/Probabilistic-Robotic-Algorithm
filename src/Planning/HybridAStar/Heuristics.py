import numpy as np
import src.Planning.HybridAStar.DubinsCurve as dc


class DubinHeuristics:
    def __init__(self, turning_radius):
        self.turning_radius = turning_radius

    """
    start_config, end_config: Start and end point of the heuristics. [x, y, theta]. """
    def heuristics(self, start_config, end_config):
        # Dubin heuristics
        dubin = dc.DubinsCurve(start_config, end_config, self.turning_radius)
        dubin_path, dubin_length = dubin.compute()

        # Euclidean heuristics
        distance = np.linalg.norm(start_config[0:2] - end_config[0:2])

        return np.fmax(dubin_length, distance)

    def precompute_dubin_heuristics(self, resolut):