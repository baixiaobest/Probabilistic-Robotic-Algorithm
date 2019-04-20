import numpy as np
import math


# grid: occupancy grid on which the raycast is done
# startPos: position vector where raycast starts.
# theta: angle of the raycast. value of [0, 2*pi].
# returns: position vector of the endpoint of the ray,
#   or an array of position vector on which the ray visited,
#   depending on the option fullPath.
def raycast(grid, startPos, theta, fullPath=False):
    path = []
    endPoint = np.zeros((2, 1))

    x = startPos[0, 0]
    y = startPos[1, 0]

    height, width = grid.shape

    if width <= x or height <= y:
        raise Exception("Start position is not in the grid")


    # Center of mass of current grid cell.
    currCmX = x + 0.5
    currCmY = y + 0.5

    deltaY = math.sin(theta)
    deltaX = math.cos(theta)

    # Calculate theta's minimum angle to 0 or PI
    thetaAngleToHorizontal = min(math.fabs(theta - 0), math.fabs(theta - math.pi), math.fabs(theta - 2 * math.pi))

    if thetaAngleToHorizontal <= 1/4.0 * math.pi:
        scale = math.fabs(1.0/deltaX)
        deltaX = deltaX * scale
        deltaY = deltaY * scale
    else:
        scale = math.fabs(1.0/deltaY)
        deltaY = deltaY * scale
        deltaX = deltaX * scale

    gridX = int(currCmX)
    gridY = int(currCmY)

    while gridX < width and gridX >=0 and gridY < height and gridY >= 0:
        # record the path if fullpath is requested.
        if fullPath:
            path.append(np.array([[gridX], [gridY]]))

        # When the current grid cell is occupied, stop the raycast.
        if grid[gridY, gridX] == 0:
            endPoint = np.array([[gridX], [gridY]])
            break

        currCmX += deltaX
        currCmY += deltaY
        # The grid position the center of mass corresponds to.
        gridX = int(currCmX)
        gridY = int(currCmY)

    if fullPath:
        return path
    else:
        return endPoint

# A Omni-directional raycast, raycast on all directions.
# Returns a list of endpoints if fullPath is false, otherwise,
#  it returns a list of rays' paths.
def raycastOmni(grid, startPos, numRays, fullPath=False):
    thetas = np.zeros(numRays)
    for i in range(numRays):
        thetas[i] = i / float(numRays) * 2 * math.pi

    pathsOrEndpoints = []
    for i in range(numRays):
        pathsOrEndpoints.append(raycast(grid, startPos, thetas[i], fullPath))

    return pathsOrEndpoints

# A raycast that simulate a laser sensor. It introduces noises into the raycast.
# Returns a vector [[theta], [Z]] where theta is angle and Z is distance.
def sensorRaycast(grid, startPos, theta, resolution=1):
    endpoint = raycast(grid, startPos, theta)
    relDist = np.linalg.norm(endpoint - startPos) * resolution
    return np.array([[theta], [relDist]])