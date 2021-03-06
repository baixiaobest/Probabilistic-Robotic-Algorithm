import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
from matplotlib import collections as mc

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
            # Occupied cell has value of 0.
            if grid[y, x] == 0:
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

'''
configs: List of rectangle configuration: [(x, y, theta), ...]
'''
def plotRectangles(configs, width, length, color=(0, 0, 1, 1)):
    lines = []
    for config in configs:
        x, y, theta = config
        front_left = [0.5 * length, 0.5 * width]
        front_right = [0.5 * length, -0.5 * width]
        rear_left = [-0.5 * length, 0.5 * width]
        rear_right = [-0.5 * length, -0.5 * width]
        vertices = [front_left, front_right, rear_right, rear_left]
        vertices_transformed = []
        for vertex in vertices:
            vx, vy = vertex
            vertices_transformed.append([
                vx * np.cos(theta) - vy * np.sin(theta) + x,
                vx * np.sin(theta) + vy * np.cos(theta) + y])

        for i in range(len(vertices_transformed)):
            next_idx = (i + 1) % len(vertices_transformed)
            lines.append([vertices_transformed[i], vertices_transformed[next_idx]])

    lc = mc.LineCollection(lines, colors=color)
    ax = plt.gca()
    ax.add_collection(lc)


'''
Plot line segments.
lines: A list of lines, defined as [[(x0, y0), (x1, y1)], [...], ...]
'''
def plotLines(lines, color=(1, 0, 0, 1)):
    lc = mc.LineCollection(lines, colors=color)
    ax = plt.gca()
    ax.add_collection(lc)

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
