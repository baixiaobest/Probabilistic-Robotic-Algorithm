import heapq as pq

class HybridAStar:
    """
    occupany_check: A class that can check occupancy given configuration [x, y, theta].
    heuristics: A class that can calculate heuristics cost given start configuration and end configuration.
    motion_model: A class that generates new configuration from current configuration.
    resolution: Resolution of the grid, {x_res, y_res, theta_res}.
    limits: Limits/bounds of the grid, {x_min, x_max, y_min, y_max}
    max_num_nodes: Maximum number of search nodes that will be explored before termination.
    """
    def __init__(self, occupany_check, heuristics, motion_model, resolutions, limits, max_num_nodes=1000):
        self.occupany_check = occupany_check
        self.heuristics = heuristics
        self.motion_model = motion_model
        self.resolutions = resolutions
        self.limits = limits
        self.max_num_nodes = int(max_num_nodes)

    """ Calculate a path from start config to end config. """
    def compute(self, start_config, end_config):
        visited_locations = {}  # visited node location on the grid. (row, col, height)
        self.nodes = []  # All the nodes generated. {config: , prev_node_id: , path: }
        first_node = {'config': start_config, 'prev_node_id': 0, 'path': None}

        # priority queue, list of to be explored nodes. Queue element: [heuristics cost + past cost, node id]
        open = []
        pq.heappush(open, [self.heuristics.heuristics(first_node), 0.0, 0])
        self.nodes.append(first_node)

        path_found = False
        while len(open) > 0 and len(self.nodes) <= self.max_num_nodes:
            [curr_est_cost, curr_node_id] = pq.heappop(open)
            curr_node_config = self.nodes[curr_node_id]['config']
            if self._is_in_same_cell(curr_node_config, end_config):
                path_found = True
                break

            # Mark current node visited.
            visited_locations.add(self._get_cell_location(curr_node_config))

            # Compute neighboring nodes.
            neighbor_configs, paths, costs = self.motion_model.generate_neighbors(curr_node_config)
            for idx, neighbor_config in enumerate(neighbor_configs):
                cell_location = self._get_cell_location(neighbor_config)
                # Neighbor is visited before, skip it.
                if cell_location in visited_locations:
                    continue
                # There is no free path to the neighbor, skip it.
                if not self.occupany_check.path_is_free(paths[idx]):
                    continue
                # Create new node
                new_node = {'config': neighbor_config, 'prev_node_id': curr_node_id, 'path': paths[idx]}
                new_node_id = len(self.nodes)
                self.nodes.append(new_node)
                # Calculate cost and push to queue
                past_cost = curr_est_cost + costs[idx]
                heuristics_cost = self.heuristics.heuristics(neighbor_config, end_config)
                pq.heappush(open, [past_cost + heuristics_cost, new_node_id])

        if not path_found:
            return False

        # Reconstruct the path.
        self.path = []
        node_id = len(self.nodes) - 1
        while not node_id == 0:
            node = self.nodes[node_id]
            self.path.append(node['path'])
            node_id = node['prev_node_id']
        self.path.reverse()

        return True

    def get_path(self):
        return self.path

    def get_path_points(self, point_interval):
        points = []
        for p in self.path:
            points = points + p.generate_points(point_interval)

    def get_all_explored_path(self):
        all_path = []
        for node in self.nodes:
            all_path.append(node['path'])
        return all_path

    def get_all_explored_path_points(self, point_interval):
        all_path = self.get_all_explored_path()
        points = []
        for p in all_path:
            points = points + p.generate_points(point_interval)
        return points

    """ Given two config, return true if they are in the same cell. """
    def _is_in_same_cell(self, config_1, config_2):
        row_1, col_1, height_1 = self._get_cell_location(config_1)
        row_2, col_2, height_2 = self._get_cell_location(config_2)
        return col_1 == col_2 and row_1 == row_2 and height_1 == height_2

    """ Given config, return its cell location. """
    def _get_cell_location(self, config):
        [x, y, theta] = config
        col = int(x - self.limits['x_min'] / self.resolutions['x_res'])
        row = int(y - self.limits['y_min'] / self.resolutions['y_res'])
        height = int(theta / self.resolutions['theta_res'])
        return row, col, height
