import random
import numpy as np


class RandomConfigSampler:
    """
    local_planner: Local planner that checks collision.
    x_limit, y_limit, theta_limit: Tuple of configuration space limit.
    """
    def __init__(self, local_planner, x_limit=None, y_limit=None, theta_limit=(0, np.pi)):
        self.local_planner = local_planner

        if x_limit is None:
            x_limit, _ = local_planner.get_xy_limit()
        if y_limit is None:
            _, y_limit = local_planner.get_xy_limit()
        self.x_limit = x_limit
        self.y_limit = y_limit

        self.theta_limit = theta_limit

    ''' Return a sampled configuration. '''
    def uniform_collision_free_sample(self):
        is_in_collision = True
        config = None

        while is_in_collision:
            config = self._sample_config()
            is_in_collision = self.local_planner.is_in_collision(config)

        return config

    ''' 
    Return a sampled configuration around the obstacle. 
        Return None if a sample is not found.
    sigma: Standard deviation for gaussian sampling. 
    trials: Number of trials to find in-collision and collision free configurations.
    '''
    def sample_around_obstacle(self, sigmas, trials=10):
        config_around_obstacle = None
        collision_trial_count = 0
        free_trial_count = 0

        while collision_trial_count < trials:
            # find a sample in collision.
            config_in_collision = self._sample_config()
            is_in_collision = self.local_planner.is_in_collision(config_in_collision)

            # find a sample not in collision.
            if is_in_collision:
                while is_in_collision and free_trial_count < trials:
                    gauss_sample_config = self._gaussian_sample_around_config(config_in_collision, sigmas)
                    is_in_collision = self.local_planner.is_in_collision(gauss_sample_config)
                    free_trial_count = free_trial_count + 1

                # A free config is found.
                if is_in_collision is False:
                    config_around_obstacle = gauss_sample_config
                    break

            collision_trial_count = collision_trial_count + 1

        return config_around_obstacle

    ''' Unifromly sample a config in the limit of the configuration space. '''
    def _sample_config(self):
        x = random.uniform(self.x_limit[0], self.x_limit[1])
        y = random.uniform(self.y_limit[0], self.y_limit[1])
        theta = random.uniform(self.theta_limit[0], self.theta_limit[1])
        return x, y, theta

    ''' Gaussian sample in the neighbourhood of the given configuration. '''
    def _gaussian_sample_around_config(self, config, sigmas):
        new_config = []
        for idx, sigma in enumerate(sigmas):
            new_config.append(np.random.normal(config[idx], sigma, 1)[0])

        return new_config

