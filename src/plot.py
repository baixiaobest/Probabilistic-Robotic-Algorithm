import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math


def plotOccupancyGrid(arr, res):
    height, width = arr.shape

    # Initial xPos, yPos size.
    N = 1000
    idx = 0
    xPos = np.zeros(N)
    yPos = np.zeros(N)
    # Loop through every pixel in the arr.
    for y in range(height):
        for x in range(width):
            if arr[y,x] < 100:
                xPos[idx] = x * res
                yPos[idx] = y * res
                idx += 1
                # Resize the array when it reaches maximum
                if idx == N:
                    N = 2 * N
                    xPos.resize(N)
                    yPos.resize(N)

    # Remove zeros in the array.
    xPos = xPos[0:idx]
    yPos = yPos[0:idx]

    # Plot the occupancy grid.
    plt.xlim(0, width * res)
    plt.ylim(0, height * res)
    plt.plot(xPos, yPos, 'k.', markersize=1)

# Plot the pose of robot.
# pose: a vector of [[x], [y], [theta]]
def plotRobotPose(pose):
    x = pose[0, 0]
    y = pose[1, 0]
    theta = pose[2, 0]
    dx = math.cos(theta)
    dy = math.sin(theta)

    plt.arrow(x, y, dx, dy)
    plt.plot([x],[y], 'bo', markersize=3)


def limit(xs, xe, ys, ye):
    plt.xlim(xs, xe)
    plt.ylim(ys, ye)


def show():
    plt.show()


def readBMPAsNumpyArray(file):
    img = Image.open("../map/maze_map.bmp")
    data = np.array((img.getdata()))
    width, height = img.size
    data = data.reshape(height, width)
    return data
