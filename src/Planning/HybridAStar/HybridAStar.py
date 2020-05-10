import heapq as pq
import numpy as np
import math
import src.Planning.HybridAStar.DubinsCurve as dcrv

class HybridAStar:
    """
    occupany_check: A class that can check occupancy given configuration [x, y, theta].
    heuristics: A class that can calculate heuristics cost given start configuration and end configuration.
    motion_model: A class that generates new configuration from current configuration.
    resolution: Resolution of the grid, {x_res, y_res, theta_res}.
    limits: Limits/bounds of the grid, {x_min, x_max, y_min, y_max}
    max_num_nodes: Maximum number of search nodes that will be explored before termination.
    bucket_interval: Nodes with different distance to the goal has different probability of performing dubin curve
        shortcut to the goal. This is implemented by putting nodes with different distance into buckets. The granularity
        of the bucket is bucket_interval. Nodes with distance difference within this interval
        generally fall into the same bucket.
    """
    def __init__(self,
                 occupany_check,
                 heuristics,
                 motion_model,
                 resolutions,
                 limits,
                 max_num_nodes=1000,
                 bucket_interval=2.0):
        self.occupany_check = occupany_check
        self.heuristics = heuristics
        self.motion_model = motion_model
        self.resolutions = resolutions
        self.limits = limits
        self.max_num_nodes = int(max_num_nodes)
        self.paths = []
        self.bucket_interval = float(bucket_interval)
        self.buckets = []

    """ Calculate a path from start config to end config. """
    def compute(self, start_config, end_config):
        visited_locations = set()  # visited node location on the grid. (row, col, height)
        self.nodes = []  # All the nodes generated. {config: , prev_node_id: , path: }
        first_node = {'config': start_config, 'prev_node_id': 0, 'path': None}

        # priority queue, list of to be explored nodes. Queue element: [heuristics cost + past cost, past_cost, node id]
        open = []
        pq.heappush(open, [self.heuristics.heuristics(start_config, end_config), 0.0, 0])
        self.nodes.append(first_node)

        self.buckets = []

        path_found = False
        while len(open) > 0 and len(self.nodes) <= self.max_num_nodes:
            [curr_est_cost, curr_past_cost, curr_node_id] = pq.heappop(open)
            curr_node_config = self.nodes[curr_node_id]['config']

            cell_location = self._get_cell_location(curr_node_config)
            # Node is visited before, skip it.
            if cell_location in visited_locations:
                continue

            # Try to connect to the goal config using Dubin's curve.
            if self._should_perform_dubin_curve(curr_node_config, end_config):
                dubin_paths, length = self._connect_using_dubin(curr_node_config, end_config)
                # A free path is found from current node to goal node.
                if self.occupany_check.paths_are_free(dubin_paths):
                    new_node = {'config': end_config, 'prev_node_id': curr_node_id, 'path': dubin_paths}
                    self.nodes.append(new_node)
                    path_found = True
                    break

            # Current node is the goal node.
            if self._is_in_same_cell(curr_node_config, end_config):
                path_found = True
                break

            # Mark current node visited.
            visited_locations.add(self._get_cell_location(curr_node_config))

            # Compute neighboring nodes.
            neighbor_configs, paths, costs = self.motion_model.generate_neighbors(curr_node_config)
            for idx, neighbor_config in enumerate(neighbor_configs):
                # There is no free path to the neighbor, skip it.
                if not self.occupany_check.paths_are_free(paths[idx]):
                    continue
                # Create new node
                new_node = {'config': neighbor_config, 'prev_node_id': curr_node_id, 'path': paths[idx]}
                new_node_id = len(self.nodes)
                self.nodes.append(new_node)
                # Calculate cost and push to queue
                past_cost = curr_past_cost + costs[idx]
                heuristics_cost = self.heuristics.heuristics(neighbor_config, end_config)
                pq.heappush(open, [past_cost + heuristics_cost, past_cost, new_node_id])

        if not path_found:
            return False

        # Reconstruct the path.
        self.paths = []
        node_id = len(self.nodes) - 1
        while not node_id == 0:
            node = self.nodes[node_id]
            node['path'].reverse()
            self.paths += node['path']
            node_id = node['prev_node_id']
        self.paths.reverse()

        return True

    """ Get the path to goal after compute is run. """
    def get_path(self):
        return self.paths

    """ Get the points on the path to goal. """
    def get_path_points(self, point_interval):
        points = []
        for p in self.paths:
            points = points + p.generate_points(point_interval)
        return self._remove_duplicate_path_points(points)

    """ Get all the paths that is explored. """
    def get_all_explored_path(self):
        all_path = []
        for node in self.nodes:
            if not node['path'] == None:
                all_path = all_path + node['path']
        return all_path

    """ Get all the points on the paths that are explored. """
    def get_all_explored_path_points(self, point_interval):
        all_path = self.get_all_explored_path()
        points = []
        for p in all_path:
            points = points + p.generate_points(point_interval)
        return points

    def _remove_duplicate_path_points(self, points):
        i = 0
        while i < len(points) - 1:
            if np.array_equal(points[i], points[i+1]):
                del points[i:i+1]
            else:
                i = i + 1
        return points



    """ Connect two configuration using dubing curve. """
    def _connect_using_dubin(self, start, end):
        min_turning_radius = self.motion_model.get_min_turning_radius()
        dubins_curve = dcrv.DubinsCurve(start, end, min_turning_radius)
        return dubins_curve.compute()

    """ Return whether or not dubin curve to goal should be calculated. Bucket counter increments."""
    def _should_perform_dubin_curve(self, config, end_config):
        distance_to_goal = np.linalg.norm(config[0:2] - end_config[0:2])

        # Each distance have corresponding bucket and bucket value.
        # Bucket value is incremented every time this function is called.
        # Dubin curve calculation is performed when this bucket value wraps back to 0.
        bucket_idx = int(distance_to_goal / self.bucket_interval)
        # Dynamically expands the bucket array if needed.
        if bucket_idx >= len(self.buckets):
            diff = bucket_idx - len(self.buckets) + 1
            self.buckets += [0] * diff
        bucket_max = bucket_idx + 1
        bucket_value = self.buckets[bucket_idx]
        # increment bucket value or wrap around.
        self.buckets[bucket_idx] = (bucket_value + 1) % bucket_max

        return bucket_value == 0

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
