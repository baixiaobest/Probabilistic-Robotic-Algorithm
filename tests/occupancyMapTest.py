import src.Models.robot as bot
import src.Models.velocityMotionModel as vm
import random
import src.Utils.plot as plot
import numpy as np
import src.Models.rangeFinderBeamModel as mm
import src.Mapping.OccupancyMap as occmap
import math
import matplotlib.pyplot as plt

command1 = [
    np.array([0.5, 0]),
    np.array([0.3, 1.7]),
    np.array([0.9, 0]),
    np.array([1, 1.5]),
    np.array([1, 1]),
    np.array([0.5, 0.45]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0.3]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 1]),
    np.array([1, 0.9]),
    np.array([1.3, 1]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, -0.6]),
    np.array([1, -0.6]),
    np.array([2, -0.3]),
]

def generateData(robot, command, initialPose):
    poses = [initialPose]
    measurements = []

    measurements.append(robot.measurementUpdate())

    for i in range(len(command)):
        poses.append(robot.motionUpdate(command[i], deltaT))
        measurements.append(robot.measurementUpdate())

    return poses, measurements

def generateLocalizationData(mcl, command, measurements, deltaT):

    estimatedPoses = []
    particles = []

    for i in range(len(command)):
        estimatedPoses.append(mcl.calculateEstimatedPose())
        particles.append(mcl.getParticles())
        mcl.motionUpdate(command[i], deltaT)
        mcl.measurementUpdate(measurements[i])
        mcl.resample()

    estimatedPoses.append(mcl.calculateEstimatedPose())
    particles.append(mcl.getParticles())

    return estimatedPoses, particles


if __name__ == '__main__':
    random.seed(0)
    grid = plot.readBMPAsNumpyArray("../map/maze_map.bmp")

    plotLim = [5, 25, 0, 25]
    initialPose = np.array([12, 18, 0])
    # 5 cm of variance in laser scanner.
    noiseSigma = 0.05
    # 1 cm per grid cell width
    resolution = 0.01
    limit = 10
    numRays = 32
    deltaT = 1

    motionModel = vm.VelocityMotionModel(
        alpha1=0.01,
        alpha2=0.005,
        alpha3=0.005,
        alpha4=0.005,
        alpha5=0.001,
        alpha6=0.001)

    mclMotionModel = vm.VelocityMotionModel(
        alpha1=0.02,
        alpha2=0.01,
        alpha3=0.02,
        alpha4=0.01,
        alpha5=0.001,
        alpha6=0.001)

    measurementModel = mm.RangeFinderBeamModel(
        w_hit=0.89,
        w_short=0.01,
        w_rand=0.05,
        w_max=0.05,
        sigma_hit=1,
        lamda_short=0.1,
        z_max=limit)

    robot = bot.Robot(grid, initialPose, motionModel, numRays, noiseSigma, resolution, limit)

    poses, measurements = generateData(robot, command1, initialPose)

    shape = (grid.shape[0] / 10, grid.shape[1] / 10)
    origin = [0, 0]
    mapResolution = 0.1
    locc = 1
    lfree = -1
    alpha = 0.5
    beta = 2 * math.pi / numRays
    map = occmap.OccupancyMap(shape, origin, mapResolution, locc, lfree, limit, alpha, beta)

    for i in range(len(poses)):
        pose = poses[i]
        measurement = measurements[i]
        map.update(pose, measurement)

    plt.figure()
    probOccMap = map.generateProbabilityOccupancyGrid(probFree=True)
    plt.imshow(probOccMap, cmap='gray', origin='lower')

    plt.figure()
    plot.plotOccupancyGrid(grid, resolution)
    plot.plotRobotPoses(poses)
    plot.show()

