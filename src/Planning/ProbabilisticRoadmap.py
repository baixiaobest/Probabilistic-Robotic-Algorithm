import src.Planning.RandomConfigSampler as rcs
import src.Planning.RoadmapNode as rn
import kdtree as kd
import heapq as pq
import numpy as np

def _kdtree_distance(node, point):
    return _distance(node.get_config(), point)

def _distance(config_start, config_end):
    left_x, left_y, left_theta = config_start
    right_x, right_y, right_theta = config_end

    dx = np.abs(left_x - right_x)
    dy = np.abs(left_y - right_y)
    dtheta = np.abs((left_theta - right_theta + np.pi) % (2 * np.pi) - np.pi)

    return np.sqrt(dx**2 + dy**2 + dtheta*2)

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
        self.kd_tree = kd.create(point_list=self.nodes, dimensions=3)

        # Check for connection for each node.
        for node in self.nodes:
            results = self.kd_tree.search_knn(node.get_config(), self.k_closest, _kdtree_distance)
            for (kdnode, distance) in results:
                neighbor_node = kdnode.data

                # If neighbor node is the current node, or
                # if distance is too large, discard it.
                if neighbor_node.get_id() == node.get_id() \
                        or distance > self.distance_upper_bound:
                    continue

                # Check connectivity. If it is connected to
                # the neighbour, add them as neighbour.
                connected = self.local_planner.check_connection(node.get_config(), neighbor_node.get_config())
                if connected:
                    node.add_neighbor(neighbor_node)

        return None

    ''' Return a list of collision free configuration from config_init to config_goal. '''
    def query(self, config_start, config_goal):
        if self.kd_tree is None:
            print("Run compute function first")
            return None

        start_node = self._find_k_closest_connected_config(config_start)
        if start_node is None:
            print("Cannot find a node in roadmap that connect to the start")
            return None

        goal_node = self._find_k_closest_connected_config(config_goal)
        if goal_node is None:
            print("Cannot find a node in roadmap that connect to the goal")
            return None

        # Perform dijkstra on the roadmap.
        # Dictionary that store previous node of a node in the path.
        path_dict = {}
        dist_dict = {}
        queue = []
        pq.heappush(queue, [0, start_node.get_id()])
        while len(queue) > 0:
            top = pq.heappop(queue)
            dist = top[0]
            curr_node = self.nodes[top[1]]

            if curr_node.get_id() == goal_node.get_id():
                break

            # Expand the search frontier to the neighboring nodes.
            neighbor_nodes = curr_node.get_neighbors()
            for neighbor_node in neighbor_nodes:
                # Caculate distance to neighbor.
                neighbor_dist = _distance(curr_node.get_config(), neighbor_node.get_config())
                total_dist = neighbor_dist + dist

                # Ignore this neighbor if this neighbor is visited
                # before and has shorter distance from other path.
                if neighbor_node.get_id() in dist_dict.keys() \
                    and total_dist > dist_dict[neighbor_node.get_id()]:
                        continue

                # add distance to the neighbor, push it into heap and record the path.
                dist_dict[neighbor_node.get_id()] = total_dist
                pq.heappush(queue, [total_dist, neighbor_node.get_id()])
                path_dict[neighbor_node.get_id()] = curr_node.get_id()

        # A path is found, reconstruct that path.
        if goal_node.get_id() in path_dict.keys():
            path = [config_start]
            path += self._reconstruct_path(path_dict, start_node.get_id(), goal_node.get_id())
            path.append(config_goal)
            return self._post_process_path(path)
        else:
            return None

    ''' Return a list of all configurations in the roadmap. '''
    def get_all_configs(self):
        return [n.get_config() for n in self.nodes]

    ''' Return a list of pair of connections. '''
    def get_all_connections(self):
        connections = []
        visited = set()
        for node in self.nodes:
            if node.get_id() in visited:
                continue

            curr_config = node.get_config()
            neighbors = node.get_neighbors()
            for neighbor_node in neighbors:
                if neighbor_node.get_id() in visited:
                    continue
                connections.append([curr_config, neighbor_node.get_config()])

            visited.add(node.get_id())

        return connections
    ''' 
    Return one of the k closest configuration node
        that is connected to given configuration.
    Return None if nothing is found.
    '''
    def _find_k_closest_connected_config(self, config):
        results = self.kd_tree.search_knn(config, self.k_closest, _kdtree_distance)
        for (kdnode, distance) in results:
            neighbor_node = kdnode.data
            neighbor_config = neighbor_node.get_config()
            if self.local_planner.check_connection(config, neighbor_config):
                return neighbor_node

        return None

    def _reconstruct_path(self, path_dict, start_id, goal_id):
        path = [goal_id]
        curr_id = goal_id
        while curr_id != start_id:
            curr_id = path_dict[curr_id]
            path.append(curr_id)

        path_config = []
        for id in path:
            path_config.append(self.nodes[id].get_config())
        path_config.reverse()

        return path_config

    def _post_process_path(self, path):
        if len(path) < 2:
            return path

        curr_idx = 0
        processed_path = []
        while curr_idx < len(path):
            lookahead_idx = curr_idx + 1
            while lookahead_idx < len(path) and \
                    self.local_planner.check_connection(path[curr_idx], path[lookahead_idx]):
                lookahead_idx += 1
            lookahead_idx -= 1
            processed_path.append(path[curr_idx])

            if lookahead_idx != curr_idx:
                curr_idx = lookahead_idx
            else:
                curr_idx += 1

        return processed_path
