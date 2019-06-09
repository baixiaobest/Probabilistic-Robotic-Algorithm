import src.Models.robotFeatureBased as r
import src.Utils.featuresGenerator as fg
import src.Utils.plot as plot
import numpy as np
import src.Models.velocityMotionModel as vm
import math
import random
import src.FastSLAM.FastSLAM_1 as slam

command = [
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([0.5, 0.2]),
    np.array([0.5, 0.5]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([0.8, 0.3]),
    np.array([0.8, 0.3]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([0.7, 0.1]),
    np.array([1, 0.1]),
    np.array([1, 0.1]),
]

# measurements are a list of [theta, dist].
def calulateFeaturesPositionsFromMeasurements(measurements, pose):
    positions = []

    for i in range(len(measurements)):
        angle = (measurements[i][0] + pose[2]) % (2 * math.pi)
        dist = measurements[i][1]
        positions.append(
            pose[0:2]
            + np.array([dist * math.cos(angle), dist * math.sin(angle)]))
    return positions

if __name__ == '__main__':
    random.seed(1)

    numFeatures = 100
    xmin = 0
    xmax = 20
    ymin = 0
    ymax = 20
    deltaT = 3

    landmarks = fg.generateFeatures(xmin, xmax, ymin, ymax, numFeatures)

    initialPose = np.array([3, 3, 0])
    detectionRange = 5
    angleNoise = 0.05
    distanceNoise = 0.1

    motionModel = vm.VelocityMotionModel(
        alpha1=0.01,
        alpha2=0.005,
        alpha3=0.005,
        alpha4=0.005,
        alpha5=0.001,
        alpha6=0.001)

    numParticles = 1
    measurementsCovariance = np.array([[angleNoise, 0], [0, distanceNoise]])

    random.seed(1)

    robot = r.robotFeatureBased(initialPose, motionModel, landmarks, detectionRange, angleNoise, distanceNoise)
    fastSlam = slam.FastSLAM_1(numParticles, initialPose, motionModel, measurementsCovariance)

    poses = [initialPose]
    measurements = []

    for i in range(len(command)):
        robot.motionUpdate(command[i], deltaT)
        poses.append(robot.getPose())

    for i in range(len(poses)):
        robot.setPose(poses[i])
        measurements.append(robot.measurementUpdate())

    for i in range(len(measurements)):
        fastSlam.getParticles()[0].setPose(poses[i])
        fastSlam.measurementUpdate(measurements[i])
        particle = fastSlam.getParticles()[0]
        fPositions = map(lambda f: f.getPosition(), particle.getFeatures())

        plot.plotRobotPose(poses[i])
        plot.plotRobotPose(particle.getPose(), style="ro")
        plot.plotFeatures(fPositions, 'bo')
        plot.plotFeatures(landmarks)
        plot.show()

        if i < len(command):
            fastSlam.motionUpdate(command[i], deltaT)
    # plot.plotRobotPoses(poses)
    # plot.plotFeatures(landmarks)
    # plot.show()
