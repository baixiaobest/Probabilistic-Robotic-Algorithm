import src.robot as bot
import src.velocityMotionModel as vm
import random
import src.plot as plot
import numpy as np

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

    motionModel = vm.VelocityMotionModel(
        alpha1=0.01,
        alpha2=0.005,
        alpha3=0.005,
        alpha4=0.005,
        alpha5=0.001,
        alpha6=0.001)

    initialPose = np.array([12, 18, 0])
    # 5 cm of variance in laser scanner.
    noiseSigma = 0.05
    # 1 cm per grid cell width
    resolution = 0.01
    limit = 10
    numRays = 8
    deltaT = 1

    robot = bot.Robot(grid, initialPose, motionModel, numRays, noiseSigma, resolution, limit)

    poses = [initialPose]
    measurements = []

    for i in range(len(command)):
        poses.append(robot.motionUpdate(command[i], deltaT))
        measurements.append(robot.measurementUpdate())

    plot.plotOccupancyGrid(grid, resolution)
    plot.plotRobotPoses(poses)
    plot.show()