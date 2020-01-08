import src.Planning.RandomConfigSampler as rcs
import src.Planning.RoadmapNode as rn
import kdtree as kd

class ProbabilisticRoadmap:
    """
    local_planner: It checks if a configuration is in collision
        or there is connection between two configurations.
    k_closest: K closest neighbours used during connection step.
    distance_upper_bound: K closest neighbours will need to be within this upper bound distance.
    num_nodes: Total number of road map nodes to generate.
    """
    def __init__(self, local_planner, k_closest, distance_upper_bound, num_nodes):
        self.local_planner = local_planner
        self.k_closest = k_closest
        self.distance_upper_bound = distance_upper_bound
        self.sampler = rcs.RandomConfigSampler(local_planner)
        self.nodes = []
        self.num_nodes = num_nodes
        self.kd_tree = None

    ''' Compute the probabilistic roadmap. '''
    def compute_roadmap(self):
        # Randomly create nodes in configuration space.
        for idx in range(self.num_nodes):
            free_config = self.sampler.uniform_collision_free_sample()
            self.nodes.append(rn.RoadmapNode(idx, free_config))

        # Construct kd tree.
        self.kd_tree = kd.create(point_list=self.nodes, dimensions=2)

        # Check for connection for each node.
        for node in self.nodes:
            results = self.kd_tree.search_knn(node, self.k_closest)
            for (kdnode, distance) in results:
                neighbor_node = kdnode.data
                # If distance is too large, discard it.
                if distance > self.distance_upper_bound:
                    continue

                # Check connectivity. If it is connected to
                # the neighbour, add them as neighbour.
                connected = self.local_planner.check_connection(node.get_config(), neighbor_node.get_config())
                if connected:
                    node.add_neighbour(neighbor_node)
                    neighbor_node.add_neighbour(node)

        return None

    ''' Return a list of collision free configuration from config_init to config_goal. '''
    def query(self, config_init, config_goal):
        return None

