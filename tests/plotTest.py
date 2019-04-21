import src.plot as plot
import src.raycast as rc
import numpy as np


def plotOccupancy():
    data = plot.readBMPAsNumpyArray("../map/maze_map.bmp")

    startPos = np.array([300, 1200])
    numRays = 8

    paths = rc.raycastOmnidirection(data, startPos, numRays, True, 1000)

    endPoints = rc.raycastOmnidirection(data, startPos, numRays, limit=3)

    for i in range(len(paths)):
        for j in range(len(paths[i])):
            x = paths[i][j][0]
            y = paths[i][j][1]
            data[y, x] = 0

    plot.plotOccupancyGrid(data, 0.01)
    plot.show()


if __name__ == '__main__':
    plotOccupancy()
