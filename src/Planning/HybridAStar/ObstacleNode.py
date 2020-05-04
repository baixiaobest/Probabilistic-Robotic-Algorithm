""" A node in the KD tree of the obstacle map. """
class ObstacleNode:
    """
    config: Node configuration, array [x, y].
    """
    def __init__(self, config):
        self.config = config
        self.neighbors = []

    def get_config(self):
        return self.config

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return self.config[i]

    def __repr__(self):
        return 'Item({}, {})'.format(self.config[0], self.config[1])