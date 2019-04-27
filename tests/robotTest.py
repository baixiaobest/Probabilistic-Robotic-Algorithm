import src.robot as bot
import src.velocityMotionModel as vm
import random
import src.plot as plot
import numpy as np
import src.MonteCarloLocalization as MCL
import src.rangeFinderBeamModel as mm

command = [
    np.array([0.5, 0]),
    np.array([0.3, 1.7]),
    np.array([0.9, 0]),
    np.array([1, 1.5]),
    np.array([1, 1]),
    np.array([0.5, 0.8]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0])

]

if __name__ == '__main__':
    random.seed(0)
    grid = plot.readBMPAsNumpyArray("../map/maze_map.bmp")

    plotLim = [5, 25, 10, 25]
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

    poses = [initialPose]
    measurements = []

    for i in range(len(command)):
        poses.append(robot.motionUpdate(command[i], deltaT))
        measurements.append(robot.measurementUpdate())

    N = 100
    poseGuess = [10, 20, 10, 20]
    estimatedPoses = []
    mcl = MCL.MonteCarloLocalization(grid, resolution, poseGuess, N, mclMotionModel, measurementModel)

    for i in range(len(command)):
        plot.plotOccupancyGrid(grid, resolution, plotLim)
        plot.plotRobotPoses(mcl.getParticles(), 'r+')
        plot.plotRobotPose(poses[i + 1], 'bo')
        plot.show()
        mcl.motionUpdate(command[i], deltaT)
        mcl.measurementUpdate(measurements[i])
        mcl.resample()

    plot.plotOccupancyGrid(grid, resolution)
    plot.plotRobotPoses(poses)
    plot.show()