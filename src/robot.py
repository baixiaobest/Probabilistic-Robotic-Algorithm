import raycast as rc
import numpy as np


'''
This class simulates a robot in given environment
defined by occupancy grid.
'''
class Robot:
    # grid: Occupancy grid.
    # initialPose: Initial pose of the robot, in [x, y, theta].
    # motionModel: A motion model that can sample new pose from previous pose.
    # numRays: Number of rays of each laser measurement.
    # noiseSigma: The sigma value of raycast sensor noise.
    # resolution: Resolution of the grid.
    # limit: Distance limit of the laser sensor.
    def __init__(self, grid, initialPose, motionModel, numRays, noiseSigma, resolution, limit):
        self.grid = grid
        self.pose = initialPose
        self.motionModel = motionModel
        self.numRays = numRays
        self.noiseSigma = noiseSigma
        self.resolution = resolution
        self.limit = limit


    # Sample a new pose from current pose.
    # ut: Command, a vector of [speed, angular speed].
    # deltaT: The duration of this motion.
    # Return a new pose.
    def motionUpdate(self, ut, deltaT):
        self.pose = self.motionModel.sampleNewPose(ut, self.pose, deltaT)
        return self.pose

    # Perform a laser measurement based on current pose.
    def measurementUpdate(self):
        measurements = rc.omniDirectionDistanceRaycast(
            self.grid, self.pose, self.numRays, self.pose[2], self.resolution, self.limit)

        # Add noise to these measurements
        for i in range(len(measurements)):
            noisyMeasurement = measurements[i] + np.random.normal(0, self.noiseSigma)
            measurements[i] = max(min(self.limit, noisyMeasurement), 0)

        return measurements


    def getPose(self):
        return self.pose