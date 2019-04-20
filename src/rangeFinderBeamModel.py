import raycast as rc
import numpy as np
from scipy.stats import norm
import math


class RangeFinderBeamModel:
    def __init__(self, w_hit, w_short, w_max, w_rand, sigma_hit, lamda_short, z_max):
        self.w_hit = w_hit
        self.w_short = w_short
        self.w_max = w_max
        self.w_rand = w_rand
        self.sigma_hit = sigma_hit
        self.lamda_short = lamda_short
        self.z_max = z_max

        sum = w_hit + w_short + w_max + w_rand
        if math.fabs(sum - 1.0) > 0.0001:
            raise Exception("w_hit, w_short, w_max, w_rand should add up to 1.")


    # Calculate the probability of sensor measurements zt, given the map and robot pose.
    # ztArr: Array of sensor measurement [[[theta], [distance]], ..., ...]
    # grid: Occupancy grid.
    # pose: Robot pose [[x], [y], [theta]].
    def calculateProb(self, ztArr, pose, grid, resolution=1):
        prob = 1
        for i in range(len(ztArr)):
            # Info from sensor measurement i
            zt = ztArr[i]
            theta = zt[0, 0]
            dist = zt[1, 0]

            # Clamp the measurement value, only [0, z_max] range is allowed.
            if dist > self.z_max:
                dist = self.z_max

            if dist < 0:
                dist = 0

            # Calculate expected measurement.
            expec_dist = rc.distanceRaycast(grid, pose, theta, resolution=resolution, limit=self.z_max)


            # We cannot have expected distance exceeds z_max,
            # otherwise, normalizer in Phit can be very big.
            if expec_dist > self.z_max:
                expec_dist = self.z_max

            prob *= self._calculateTotalProb(dist, expec_dist)

        return prob


    def _calculateTotalProb(self, dist, expec_dist):
        # Calculate probabilities.
        Phit = self._calculatePhit(dist, expec_dist)
        Pshort = self._calculatePshort(dist, expec_dist)
        Pmax = self._calculatePmax(dist)
        Prand = self._calculatePrand(dist)

        # Sum up all the probabilities.
        return self.w_hit * Phit + self.w_short * Pshort + self.w_max * Pmax + self.w_rand * Prand


    # Calculate Phit probability of sensor given distance and expected distance.
    def _calculatePhit(self, dist, expec_dist):

        if dist > self.z_max or dist < 0:
            return 0

        prob = norm.pdf(dist, loc=expec_dist, scale=self.sigma_hit)

        # If the measured distance and expected distance are 3 standard deviation away,
        # then we treat the normalizer as 1.
        if math.fabs(expec_dist - dist) >= 3 * self.sigma_hit:
            normalizer = 1.0
        else:
            normalizer = 1.0 / (
                norm.cdf(self.z_max, loc=expec_dist, scale=self.sigma_hit)
                - norm.cdf(0, loc=expec_dist, scale=self.sigma_hit))

        return prob * normalizer

    def _calculatePshort(self, dist, expec_dist):

        if dist > expec_dist or dist < 0:
            return 0

        # Avoid division by zero.
        if expec_dist <= 0:
            expec_dist = 0.1

        prob = self.lamda_short * math.exp(-self.lamda_short * dist)
        normalizer = 1.0 / (1.0 - math.exp(-self.lamda_short * expec_dist))

        return prob * normalizer

    def _calculatePmax(self, dist):
        if dist >= self.z_max:
            return 1.0
        else:
            return 0

    def _calculatePrand(self, dist):

        if dist < 0 or dist > self.z_max:
            return 0

        return 1.0 / self.z_max