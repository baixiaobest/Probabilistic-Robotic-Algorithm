import numpy as np

'''
Class that generates potential field given obstacle map.
'''
class PotentialField:
    '''
    map: Numpy array, 2d or 3d map of obstacles. 1 means obstacle, otherwise, free space.
    limits: Dictionary of tuples {xlim: (min, max), ylim: (c, d), zlim: (e, f)}
    params: Dictionary of potential field parameters: zeta ζ, d_goal, eta η, q_star.
    resolution: Configuration space resolution, unit/pixel.
    topology: Topological space the map represents, could be
        Torus or Euclidean.
    dimension: Either 2 or 3.
    connectivity: For a cell, under what condition is the adjacent cell considered
        connected to this cell. Connected by 'edge' or 'vertex'.
    '''
    def __init__(self, obstacle_map, params, resolution=1.0, topology='euclidean', dimension=2, connectivity='edge'):
        self.obstacle_map = obstacle_map
        self.params = params
        self.resolution = resolution
        self.topology = topology
        self.dimension = dimension
        self.connectivity = connectivity
        self.obstacle_dist_map = None
        self.repulsive_potential_field = None
        self.attractive_potential_field = None
        self.negative_gradient_field_x = None
        self.negative_gradient_field_y = None
        self.OBSTACLE_NUM = 0

        if dimension != 2:
            raise NotImplemented("Dimension other than 2 is not implemented!")

    '''
    Generate a map, each cell containing minimum manhattan distance to adjacent obstacle
    '''
    def _compute_obstacle_dist_map(self):
        h, w = self.obstacle_map.shape

        # -1 means uninitialized distance value.
        UNINITIALIZED_VAL = -1
        self.obstacle_dist_map = np.full((h, w), UNINITIALIZED_VAL)

        # Add obstacles to frontier.
        frontier = []
        for row in range(h):
            for col in range(w):
                # Obstacle.
                if self.obstacle_map[row, col] == self.OBSTACLE_NUM:
                    frontier.append((row, col))
                    self.obstacle_dist_map[row, col] = 0

        while len(frontier) > 0:
            row, col = frontier.pop(0)
            dist = self.obstacle_dist_map[row, col]

            neighbors = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
            if self.connectivity == 'vertex':
                neighbors = neighbors + [(row + 1, col + 1), (row + 1, col - 1), (row - 1, col - 1), (row - 1, col + 1)]

            # For each neighbor, calculate distance.
            for n in neighbors:
                n_row, n_col = n
                # Euclidean topology cannot wrap around.
                if self.topology == 'euclidean':
                    if n_row < 0 or n_row >= h or n_col < 0 or n_col >= w:
                        break
                # Torus topology can wrap around.
                elif self.topology == 'torus':
                    n_row = (n_row + h) % h
                    n_col = (n_col + w) % w

                # If neighbor is uninitialized or having larger obstacle distance values,
                # update that value, and put it into frontier.
                if self.obstacle_dist_map[n_row, n_col] == UNINITIALIZED_VAL \
                    or dist + 1 < self.obstacle_dist_map[n_row, n_col]:
                    self.obstacle_dist_map[n_row, n_col] = dist + 1
                    frontier.append((n_row, n_col))

    '''
    Generate a potential field as a result of presence of obstacle.
    '''
    def _compute_repulsive_potential_field(self):
        if self.obstacle_dist_map is None:
            self._compute_obstacle_dist_map()

        self.repulsive_potential_field = np.zeros(self.obstacle_dist_map.shape)

        h, w = self.obstacle_dist_map.shape

        for row in range(h):
            for col in range(w):
                dist = self.obstacle_dist_map[row, col] * self.resolution
                potential = 0

                if dist == 0:
                    potential = self.params['max_potential']
                elif dist <= self.params['q_star']:
                    potential = 0.5 * self.params['eta'] * (1.0 / dist - 1.0 / self.params['q_star']) ** 2

                potential = min(self.params['max_potential'], potential)

                self.repulsive_potential_field[row, col] = potential

    '''
    Generate attractive potential field due to presence of goal.
    '''
    def _compute_attractive_potential_field(self, goal):
        h, w = self.obstacle_map.shape
        self.attractive_potential_field = np.zeros((h, w))

        x_goal, y_goal = goal

        for row in range(h):
            for col in range(w):
                x_curr = col * self.resolution
                y_curr = row * self.resolution

                if self.topology == 'torus':
                    x_diff = min(np.fabs(x_curr - x_goal), np.fabs(x_curr + w * self.resolution - x_goal))
                    y_diff = min(np.fabs(y_curr - y_goal), np.fabs(y_curr + h * self.resolution - y_goal))
                else:
                    x_diff = x_curr - x_goal
                    y_diff = y_curr - y_goal
                distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

                zeta = self.params['zeta']
                d_goal = self.params['d_goal']

                if distance <= self.params['d_goal']:
                    potential = 0.5 * zeta * distance ** 2
                else:
                    potential = d_goal * zeta * distance - 0.5 * zeta * d_goal ** 2
                self.attractive_potential_field[row, col] = potential

    def get_potential_field(self):
        return self.get_repulsive_potential_field() + self.get_attractive_potential_field()

    def get_obstacle_distance_map(self):
        if self.obstacle_dist_map is None:
            self._compute_obstacle_dist_map()
        return self.obstacle_dist_map

    def get_repulsive_potential_field(self):
        if self.repulsive_potential_field is None:
            self._compute_repulsive_potential_field()
        return self.repulsive_potential_field

    def get_attractive_potential_field(self):
        if self.goal is None:
            raise Exception("You need to set goal using set_goal function.")

        if self.attractive_potential_field is None:
            self._compute_attractive_potential_field(self.goal)

        return self.attractive_potential_field

    '''
    config: Current configuration.
    return: Gradient vector
    '''
    def get_negative_gradient(self, config):
        potential_field = self.get_potential_field()
        h, w = potential_field.shape

        x, y = config
        col, row = int(x / self.resolution), int(y / self.resolution)

        # A ring of pixels surrounding current configuration.
        neighbor_pixels = [(col, row - 3), (col + 1, row - 3), (col + 2, row - 2), (col + 3, row - 1), (col + 3, row), \
                          (col + 3, row + 1), (col + 2, row + 2), (col + 1, row + 3), (col, row + 3), (col - 1, row + 3),\
                          (col - 2, row + 2), (col - 3, row + 1), (col - 3, row), (col - 3, row - 1), (col - 2, row - 2), \
                          (col - 1, row - 3)]

        min_potential = np.Inf
        min_pixel = None

        # Filter through these neighboring pixels,
        # out of range pixels will not be considered in the next step.
        for n in neighbor_pixels:
            col_n, row_n = n
            if self.topology == 'torus':
                col_n = (col_n + w) % w
                row_n = (row_n + h) % h

            if col_n < 0 or col_n >= w or row_n < 0 or row_n >= h:
                continue

            # This pixel is valid, find minimum potential.
            if potential_field[row_n, col_n] < min_potential:
                min_potential = potential_field[row_n, col_n]
                min_pixel = n

        # Compute gradient vector.
        col_min, row_min = min_pixel
        x_min, y_min = col_min * self.resolution, row_min * self.resolution
        x_diff = x_min - x
        y_diff = y_min - y
        diff_vector = np.array([x_diff, y_diff])
        diff_norm = np.linalg.norm(diff_vector)
        direction = diff_vector / diff_norm
        potential_diff = potential_field[row, col] - potential_field[row_min, col_min]

        gradient_mag = potential_diff / diff_norm
        if gradient_mag > self.params["max_gradient"]:
            gradient_mag = self.params["max_gradient"]
        gradient = direction * gradient_mag

        return gradient

    '''
    Return an NxN grid of 2d vectors.
    '''
    def get_negative_gradient_field(self):
        if self.negative_gradient_field_x is not None and self.negative_gradient_field_y is not None:
            return self.negative_gradient_field_x, self.negative_gradient_field_y

        h, w = self.obstacle_map.shape
        self.negative_gradient_field_x = np.zeros((h, w))
        self.negative_gradient_field_y = np.zeros((h, w))

        for row in range(h):
            for col in range(w):
                gradient = self.get_negative_gradient((col * self.resolution, row * self.resolution))
                self.negative_gradient_field_x[row, col] = gradient[0]
                self.negative_gradient_field_y[row, col] = gradient[1]

        return self.negative_gradient_field_x, self.negative_gradient_field_y
    '''
    goal: Goal in configuration space.
    '''
    def set_goal(self, goal):
        self.goal = goal
        # Reset attractive potential field because the goal changes.
        self.attractive_potential_field = None
        self.negative_gradient_field_x = None
        self.negative_gradient_field_y = None