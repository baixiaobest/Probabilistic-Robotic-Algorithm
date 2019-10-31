import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math


def plotOccupancyGrid(grid, resolution, lim=None):
    height, width = grid.shape

    # Initial xPos, yPos size.
    N = 1000
    idx = 0
    xPos = np.zeros(N)
    yPos = np.zeros(N)
    # Loop through every pixel in the arr.
    for y in range(height):
        for x in range(width):
            if grid[y, x] < 100:
                xPos[idx] = x * resolution
                yPos[idx] = y * resolution
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
    if lim is None:
        plt.xlim(0, width * resolution)
        plt.ylim(0, height * resolution)
    else:
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])
    plt.plot(xPos, yPos, 'k.', markersize=1)

# Plot the pose of robot.
# pose: a vector of [x, y, theta]
def plotRobotPose(pose, style='bo'):
    x = pose[0]
    y = pose[1]
    theta = pose[2]
    dx = math.cos(theta)
    dy = math.sin(theta)

    plt.arrow(x, y, dx, dy)
    plt.plot([x],[y], style, markersize=3)

# Plot a list of robot poses.
# poses: A list of poses, or [[x, y, theta], [...], ...]
def plotRobotPoses(poses, style='bo'):
    for i in range(len(poses)):
        plotRobotPose(poses[i], style)

def plotFeatures(features, style='r+'):
    xlist = []
    ylist = []
    for i in range(len(features)):
        xlist.append(features[i][0])
        ylist.append(features[i][1])
    plt.plot(xlist, ylist, style)

def limit(xs, xe, ys, ye):
    plt.xlim(xs, xe)
    plt.ylim(ys, ye)


def show():
    plt.show()


def readBMPAsNumpyArray(file):
    img = Image.open(file)
    data = np.array((img.getdata()))
    width, height = img.size
    data = data.reshape(height, width)
    return data
