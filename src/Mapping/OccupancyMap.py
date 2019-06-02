import numpy as np
import math

'''
Class that generates occupancy grid map given robot pose and laser measurements.
'''
class OccupancyMap:
    # mapSize: Size of map in pixels, using numpy shape (horizontal_size, vertical_size).
    # origin: Origin of this map in pixel position, [px, py], this origin is where we consider as robot pose [0, 0]
    # resolution: Resolution of the map, in meter/pixel.
    # locc: Log probability of cell being occupied over being free.
    # lfree: Log probability of cell being free over being occupied.
    # zmax: Maximum range of laser.
    # alpha: alpha is the thickness of the wall in meters.
    # beta: beta is the angular region of each laser is responsible for.
    def __init__(self, mapSize, origin, resolution, locc, lfree, zmax, alpha, beta):
        # Occupancy probability map.
        self.probMap = np.zeros(mapSize)
        self.origin = origin
        self.resolution = resolution
        self.locc = locc
        self.lfree = lfree
        self.zmax = zmax
        self.alpha = alpha
        self.beta = beta

    # This function updates the probability map.
    # pose: The pose of the robot [x, y, theta]
    # measurements: A list of laser range measurements, [[theta, distance], [...], ...] when robot is at pose.
    def update(self, pose, measurements):
        width = self.probMap.shape[0]
        height = self.probMap.shape[1]

        zmaxPixels = int(math.ceil(self.zmax / self.resolution))

        poseInPixel = [int(pose[0] / self.resolution), int(pose[1] / self.resolution)]

        pxMin = max(poseInPixel[0] - zmaxPixels + self.origin[0], 0)
        pxMax = min(poseInPixel[0] + zmaxPixels + self.origin[0], width)
        pyMin = max(poseInPixel[1] - zmaxPixels + self.origin[1], 0)
        pyMax = min(poseInPixel[1] + zmaxPixels + self.origin[1], height)

        # Loop through every pixel in the probability grid.
        for py in range(pyMin, pyMax):
            for px in range(pxMin, pxMax):
                pixelPose = np.array([(px - self.origin[0]) * self.resolution, (py - self.origin[1]) * self.resolution])
                relPos = pixelPose - pose[0:2]
                relDist = math.sqrt(relPos[0] ** 2 + relPos[1] ** 2)

                if relDist > self.zmax:
                    continue

                # Angle between pixel and robot in global coordinate, in range [0, 2*pi].
                phi = (math.atan2(relPos[1], relPos[0]) + 2 * math.pi) % (2 * math.pi)

                # Find the laser measurement that corresponds to this pixel.
                idx = 0
                angleDiff = 10 # An arbitrarily impossible number.
                for i in range(len(measurements)):
                    # Calculate angle between theta and measurement.
                    diff = math.fabs((phi - measurements[i][0] + math.pi) % (2 * math.pi) - math.pi)
                    if diff < angleDiff:
                        angleDiff = diff
                        idx = i

                # Most likely measurement that corresponds to the pixel [px, py]
                zt = measurements[idx]
                # If pixel is out of measurement range or is not within a certain angle of the laser beam, ignore it.
                if relDist > zt[1] + self.alpha / 2 or angleDiff > self.beta / 2:
                    continue
                # When this pixel corresponds to an obstacle.
                if zt[1] < self.zmax and math.fabs(relDist - zt[1]) < self.alpha / 2.0:
                    self.probMap[py, px] += self.locc
                # When this pixel corresponds to free space.
                if relDist <= zt[1]:
                    self.probMap[py, px] += self.lfree

    # Return a probability grid with probability in [0, 1], probability represents the prob of cell being occupied.
    # probFree: The values in map represent the probability of it being
    # free, instead of probability of it being occupied.
    def generateProbabilityOccupancyGrid(self, probFree=False):
        shape = self.probMap.shape
        occupancyGrid = np.zeros(shape)
        height = shape[0]
        width = shape[1]

        for py in range(height):
            for px in range(width):
                if probFree:
                    occupancyGrid[py, px] = 1.0 / (1 + np.exp(self.probMap[py, px]))
                else:
                    occupancyGrid[py, px] = 1.0 - 1.0 / (1 + np.exp(self.probMap[py, px]))

        return occupancyGrid
