import plot
import raycast as rc
import numpy as np
import velocityMotionModel as vm

def plotRobotMotion():
    N = 100
    control = np.array([[1], [0.2]])
    pose = np.array([[15], [15], [0]])
    deltaT = 5

    plot.limit(0, 30, 0, 30)
    plot.plotRobotPose(pose)

    model = vm.VelocityMotionModel(0.001, 0.001, 0.05, 0.05, 0.001, 0.001)
    poses = []
    for i in range(N):
        newPose = model.sampleNewPose(control, pose, deltaT)
        poses.append(newPose)
        plot.plotRobotPose(newPose)
    plot.show()

def plotOccupancy():
    data = plot.readBMPAsNumpyArray("../map/maze_map.bmp")

    startPos = np.array([[1575], [850]])
    numRays = 8

    paths = rc.raycastOmni(data, startPos, numRays, True)

    endPoints = rc.raycastOmni(data, startPos, numRays)

    for i in range(len(paths)):
        for j in range(len(paths[i])):
            x = paths[i][j][0, 0]
            y = paths[i][j][1, 0]
            data[y, x] = 0

    plot.plotOccupancyGrid(data, 0.01)
    plot.show()

if __name__ == "__main__":
    plotRobotMotion()