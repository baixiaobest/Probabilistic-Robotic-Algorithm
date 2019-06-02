import src.Utils.plot as plot
import src.Utils.raycast as rc
import numpy as np
import matplotlib.pyplot as plt
import math


def plotOccupancy():
    data = plot.readBMPAsNumpyArray("../map/maze_map.bmp")

    startPos = np.array([300, 1200])
    # startPos = np.array([1200, 1800])
    numRays = 8

    paths = rc.raycastOmnidirection(data, startPos, numRays, True, 1000)

    endPoints = rc.raycastOmnidirection(data, startPos, numRays, limit=3)

    for i in range(len(paths)):
        for j in range(len(paths[i])):
            x = paths[i][j][0]
            y = paths[i][j][1]
            data[y, x] = 0

    plot.plotOccupancyGrid(data, 0.01)


def plotDistances():
    data = plot.readBMPAsNumpyArray("../map/maze_map.bmp")
    pose = np.array([3, 12])
    # pose = np.array([12, 18])
    numRays = 8
    distances = rc.omniDirectionDistanceRaycast(data, pose, numRays, 0, 0.01, 10)
    plt.plot(np.linspace(0, 2*math.pi, numRays), distances)


if __name__ == '__main__':
    plt.figure()
    plotOccupancy()
    plt.figure()
    plotDistances()
    plt.show()
