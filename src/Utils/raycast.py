import numpy as np
import math
from enum import Enum

# This tells why the raycast stops.
class RaycastStatus(Enum):
    # Raycast reaches the boundary of the occupancy grid.
    BOUNDARY_REACHED = 1,
    # Raycast ends because it reaches an obstacle.
    OBSTACLE = 2,
    # Raycast ends because it reaches the range limit.
    LIMIT_REACHED = 3


# grid: Occupancy grid on which the raycast is done
# startPos: Position vector where raycast starts. Numpy [x, y]
# theta: Angle of the raycast. value of range [0, 2*pi].
# fullPath: If True, then a full path of raycast will be returned.
# limit: If defined positive, raycast stops when it reaches maximum distance. Unit is per cell width.
# returns: position vector of the endpoint of the ray,
#   or an array of position vector on which the ray visited,
#   depending on the option fullPath.
#   And the status of the raycast.
def raycast(grid, startPos, theta, fullPath=False, limit=-1.0):
    path = []
    endPoint = np.zeros((2, 1))

    x = startPos[0]
    y = startPos[1]

    height, width = grid.shape

    if width <= x or height <= y:
        raise Exception("Start position is not in the grid")


    # Center of mass of current grid cell.
    currCmX = x + 0.5
    currCmY = y + 0.5

    deltaY = math.sin(theta)
    deltaX = math.cos(theta)

    # Calculate the minimum angle between the ray and the horizontal axis.
    thetaAngleToHorizontal = min(math.fabs(theta - 0), math.fabs(theta - math.pi), math.fabs(theta - 2 * math.pi))

    if thetaAngleToHorizontal <= 1/4.0 * math.pi:
        # Here, deltaX becomes a unit length.
        # So during raycast, we increment position x by 1 at a time.
        scale = math.fabs(1.0/deltaX)
        deltaX = deltaX * scale
        deltaY = deltaY * scale
    else:
        # Here, deltaY becomes a unit length.
        # So during raycast, we increment position y by 1 at a time.
        scale = math.fabs(1.0/deltaY)
        deltaY = deltaY * scale
        deltaX = deltaX * scale

    gridX = int(currCmX)
    gridY = int(currCmY)
    raycastDistance = 0.0
    raycastStatus = RaycastStatus.BOUNDARY_REACHED

    while gridX < width and gridX >=0 and gridY < height and gridY >= 0:

        # record the path if full path is requested.
        if fullPath:
            path.append(np.array([gridX, gridY]))

        # When the current grid cell is occupied, stop the raycast.
        if grid[gridY, gridX] == 0:
            endPoint = np.array([gridX, gridY])
            raycastStatus = RaycastStatus.OBSTACLE
            break

        # When maximum raycast distance limit is reached, stop the raycast.
        if limit > 0 and raycastDistance >= limit:
            endPoint = np.array([gridX, gridY])
            raycastStatus = RaycastStatus.LIMIT_REACHED
            break

        currCmX += deltaX
        currCmY += deltaY

        # The grid position the center of mass corresponds to.
        gridX = int(currCmX)
        gridY = int(currCmY)

        # It happens that the scale is the distance the ray traverse
        # during each iteration.
        raycastDistance += scale

    # The raycast ends because it reaches the grid boundary.
    # Readjust gridX and gridY to be used as endPoint.
    if gridX >= width or gridX < 0 or gridY >= height or gridY < 0:
        gridX = min(max(gridX, 0), width - 1)
        gridY = min(max(gridY, 0), height - 1)
        endPoint = np.array([gridX, gridY])

    if fullPath:
        return path, raycastStatus
    else:
        return endPoint, raycastStatus

# A Omni-directional raycast, raycast on all directions.
# startPos: Start position of the raycast, numpy [x, y]
# numRays: Total number of rays to cast in all directions.
# fullPath: Is the function returning a full path of raycast or just the endpoint.
# limit: Distance limit of the raycast, in the unit of cell width.
# Returns a list of endpoints if fullPath is false, otherwise,
#  it returns a list of rays' paths.
def raycastOmnidirection(grid, startPos, numRays, fullPath=False, limit=-1.0):
    thetas = np.zeros(numRays)
    for i in range(numRays):
        thetas[i] = i / float(numRays) * 2 * math.pi

    pathsOrEndpoints = []
    for i in range(numRays):
        pathsOrEndpoints.append(raycast(grid, startPos, thetas[i], fullPath, limit)[0])

    return pathsOrEndpoints

# A raycast that returns only distance in robot coordinate
# Grid: Occupancy grid.
# pose: Robot pose numpy [x, y].
# theta: Direction at which the ray is cast.
# resolution: Resolution of the grid, unit of meter per cell width.
# limit: Distance limit of the raycast, unit of meter.
# returnVector: Return the result in form of [theta, distance]
# Returns a distance starting from pose to the end of raycast, unit of meter.
#   Or in form of [theta, distance]
def distanceRaycast(grid, pose, theta, resolution=1, limit=-1.0, returnVector=False):
    # Transform robot pose into start position on the grid.
    startPos = np.array([
        int(pose[0] / resolution),
        int(pose[1] / resolution)])

    # Transform distance in robot coordinate into distance in grid coordinate
    gridLimit = limit / resolution

    endpoint, status = raycast(grid, startPos, theta, limit=gridLimit)

    # If raycast ends due to reaching the boundary of the occupancy grid,
    # Then this raycast has distance equals to limit.
    if status == RaycastStatus.BOUNDARY_REACHED and limit > 0:
        relativeDist = limit
    else:
        relativeDist = np.linalg.norm(endpoint - startPos) * resolution

    if returnVector:
        return np.array([theta, relativeDist])
    else:
        return relativeDist

# A Omni-directional raycast on all directions, it returns distance to surrounding
# objects only. If the raycast limit is reached, function returns that limit.
# Grid: Occupancy grid.
# pose: Rangfinder pose numpy [x, y].
# numRays: Total number of rays to cast in all directions.
# thetaOffset: Angle offset of this raycast.
# resolution: Resolution of the grid, unit of meter per cell width.
# returnVector: Return the result in form of [[theta, distance], [...], ...]
# limit: Distance limit of the raycast, unit of meter.
# Returns a list of distances. Or a list of measurement vector.
def omniDirectionDistanceRaycast(grid, pose, numRays, thetaOffset, resolution=1, limit=-1.0, returnVector=False):
    thetas = np.zeros(numRays)
    for i in range(numRays):
        thetas[i] = (i / float(numRays) * 2 * math.pi + thetaOffset) % (2 * math.pi)

    distances = []
    for i in range(numRays):
        distances.append(distanceRaycast(grid, pose, thetas[i], resolution, limit, returnVector))

    return distances