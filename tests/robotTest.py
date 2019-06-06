import src.Models.robot as bot
import src.Models.velocityMotionModel as vm
import random
import src.Utils.plot as plot
import numpy as np
import src.Localization.MonteCarloLocalization as MCL
import src.Models.rangeFinderBeamModel as mm

command1 = [
    np.array([0.5, 0]),
    np.array([0.3, 1.7]),
    np.array([0.9, 0]),
    np.array([1, 1.5]),
    np.array([1, 1]),
    np.array([0.5, 1]),
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
    # np.array([1, 0]),
    # np.array([1, 0]),
    # np.array([1, 1]),
    # np.array([1, 0.7]),
    # np.array([1.3, 0.6]),
    # np.array([1, 0.5]),
    # np.array([1, 0]),
    # np.array([1, 0]),
    # np.array([1, 0]),
    # np.array([1, -0.6]),
    # np.array([1, -0.6]),
    # np.array([2, -0.3]),
]

command2 = [
    np.array([1, -0.2]),
    np.array([0.7, 0.2]),
    np.array([1.2, -0.5]),
    np.array([1, 0.8]),
    np.array([0.8, 0.4]),
    np.array([1.5, 0.3]),
    np.array([1.2, 0]),
    np.array([1, -0.5]),
    np.array([1, 0]),
    np.array([1.2, 0]),
]

def generateData(robot, command, initialPose):
    poses = [initialPose]
    measurements = []

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
    numRays = 8
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

    # Robot is delocalized to newPose
    newPose = np.array([20, 20, 3.14])
    robot = bot.Robot(grid, newPose, motionModel, numRays, noiseSigma, resolution, limit)
    newPoses, newMeasurements = generateData(robot, command2, newPose)
    # Remove first pose to match commands array dimension
    newPoses.pop(0)
    poses.extend(newPoses)
    measurements.extend(newMeasurements)
    command1.extend(command2)


    # We use different random seed to generate localization particles
    random.seed(1)

    N = 500
    poseGuess = [8, 22, 3, 23]
    mcl = MCL.MonteCarloLocalization(grid,
                                     resolution,
                                     poseGuess,
                                     N,
                                     mclMotionModel,
                                     measurementModel,
                                     enableParticleInjection=True,
                                     alphaFast=1.2,
                                     alphaSlow=0.8)

    estimatedPoses, particles = generateLocalizationData(mcl, command1, measurements, deltaT)

    # for i in range(0, len(estimatedPoses)):
    #     plot.plotOccupancyGrid(grid, resolution, plotLim)
    #     plot.plotRobotPoses(particles[i], 'r+')
    #     plot.plotRobotPose(poses[i], 'bo')
    #     plot.show()

    plot.plotOccupancyGrid(grid, resolution)
    plot.plotRobotPoses(poses)
    plot.plotRobotPoses(estimatedPoses, 'r+')
    plot.show()