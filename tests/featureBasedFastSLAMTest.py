import src.Models.robotFeatureBased as r
import src.Utils.featuresGenerator as fg
import src.Utils.plot as plot
import numpy as np
import src.Models.velocityMotionModel as vm
import math
import random

def calulateFeaturesPositionsFromMeasurements(measurements, pose):
    positions = []

    for i in range(len(measurements)):
        angle = measurements[i][0]
        dist = measurements[i][1]
        positions.append(
            pose[0:1]
            + np.array([dist * math.cos(angle), dist * math.sin(angle)]))
    return positions

if __name__ == '__main__':
    random.seed(0)

    numFeatures = 100
    xmin = 0
    xmax = 100
    ymin = 0
    ymax = 100

    landmarks = fg.generateFeatures(xmin, xmax, ymin, ymax, numFeatures)

    initialPose = np.array([50, 50, 0])
    detectionRange = 10
    angleNoise = 0.1
    distanceNoise = 2

    motionModel = vm.VelocityMotionModel(
        alpha1=0.01,
        alpha2=0.005,
        alpha3=0.005,
        alpha4=0.005,
        alpha5=0.001,
        alpha6=0.001)

    robot = r.robotFeatureBased(initialPose, motionModel, landmarks, detectionRange, angleNoise, distanceNoise)

    measurements = robot.measurementUpdate()

    features = calulateFeaturesPositionsFromMeasurements(measurements, initialPose)

    plot.plotFeatures(landmarks)
    plot.plotFeatures(features, 'bo')
    plot.show()