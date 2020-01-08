class RoadmapNode:
    def __init__(self, id, config):
        self.id = id
        self.config = config
        self.neighbors = []

    def get_config(self):
        return self.config

    def get_id(self):
        return self.id

    ''' Add another RoadmapNode as neighbor. '''
    def add_neighbor(self, node):
        self.neighbors.append(node)

    def get_neighbors(self):
        return self.neighbors

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return self.config[0:2]

    def __repr__(self):
        return 'Item({}, {})'.format(self.config[0], self.config[1])