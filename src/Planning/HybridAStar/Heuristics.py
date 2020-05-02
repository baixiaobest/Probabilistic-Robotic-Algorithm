import src.Planning.HybridAStar.DubinsCurve as dc


class DubinHeuristics:
    def __init__(self, turning_radius):
        self.turning_radius = turning_radius

    def heuristics(self, start_config, end_config):
        dubin = dc.DubinsCurve(start_config, end_config, self.turning_radius)
        path, length = dubin.compute()
        return length