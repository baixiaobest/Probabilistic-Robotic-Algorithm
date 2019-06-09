import math
import src.Utils.sampler as s
import numpy as np


class robotFeatureBased:
    # initialPose: Initial position & orientation of the robot, in [x, y, theta].
    # motionModel: A motion model that can sample new pose from previous pose.
    # features: A list of features detectable by robot. A feature is np array of [x, y]
    # detectionRange: Range of robot detection.
    # angleNoise: Sigma of noise added to angle measurement of the feature.
    # distanceNoise: Sigma of noise added to distance measurement of the feature.
    def __init__(self, initialPose, motionModel, features, detectionRange, angleNoise, distanceNoise):
        self.pose = initialPose
        self.motionModel = motionModel
        self.features = features
        self.detectionRange = detectionRange
        self.angleNoise = angleNoise
        self.distanceNoise = distanceNoise

    # Sample a new pose from current pose.
    # ut: Command, a vector of [speed, angular speed].
    # deltaT: The duration of this motion.
    # Return a new pose.
    def motionUpdate(self, ut, deltaT):
        self.pose = self.motionModel.sampleNewPose(ut, self.pose, deltaT)
        return self.pose

    # Perform measurement on surrounding features.
    # It returns a list of features' bearings and distances if these features are within detection range.
    # Bearings are relative bearings to robot orientation.
    def measurementUpdate(self):
        detectionRangeSq = self.detectionRange ** 2
        measurements = []

        for i in range(len(self.features)):
            dx = self.features[i][0] - self.pose[0]
            dy = self.features[i][1] - self.pose[1]
            distSq = (dx) ** 2 + (dy) ** 2
            if distSq < detectionRangeSq:
                dist = math.sqrt(distSq)
                measuredDist = min(s.sampleNormal(dist, self.distanceNoise), self.detectionRange)
                angle = math.atan2(dy, dx) % (2 * math.pi)
                measuredAngle = (s.sampleNormal(angle, self.angleNoise) - self.pose[2]) % (2 * math.pi)
                measurements.append(np.array([measuredAngle, measuredDist]))

        return measurements

    def getPose(self):
        return self.pose

    def setPose(self, pose):
        self.pose = pose