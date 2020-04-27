import numpy as np
import src.Planning.HybridAStar.DubinCircle as dc
import math

class CircularPath:
    def __init__(self, position, start_theta, end_theta, circle_type, radius):
        self.position = position
        self.start_theta = start_theta
        self.end_theta = end_theta
        self.cirle_type = circle_type
        self.raidus = radius

class StraightPath:
    def __init__(self, start_position, end_position):
        self.start_position = start_position
        self.end_position = end_position

class DubinsCurve:
    """
    start: start configuration in [x, y, theta]
    end: end configuration.
    """
    def __init__(self, start, end, turning_radius):
        self.start = np.array(start).astype(float)
        self.end = np.array(end).astype(float)
        self.turning_radius = turning_radius

    def compute(self):
        # find CSC solution

        # find CCC solution
        pass


    def _find_CSC_solution(self):
        # Two circles tangent to the starting position.
        # On left and right side of the starting position.
        start_theta = self.start[2]
        dtheta = np.pi / 2
        start_ccw_circle_position = self.start[0:2] \
                                    + self.turning_radius * np.array([np.cos(start_theta + dtheta),
                                                                      np.sin(start_theta + dtheta)])
        start_cw_circle_position = self.start[0:2] \
                                    + self.turning_radius * np.array([np.cos(start_theta - dtheta),
                                                                      np.sin(start_theta - dtheta)])
        # Two circles tangent to the ending position.
        # On left and right side of the ending position.
        end_theta = self.end[2]
        end_ccw_circle_position = self.end[0:2] \
                                    + self.turning_radius * np.array([np.cos(end_theta + dtheta),
                                                                      np.sin(end_theta + dtheta)])
        end_cw_circle_position = self.end[0:2] \
                                  + self.turning_radius * np.array([np.cos(end_theta - dtheta),
                                                                    np.sin(end_theta - dtheta)])
        start_ccw_circle = dc.DubinCircle(start_ccw_circle_position, dc.CircleType.COUNTER_CLOCKWISE)
        start_cw_circle = dc.DubinCircle(start_cw_circle_position, dc.CircleType.CLOCKWISE)
        end_ccw_circle = dc.DubinCircle(end_ccw_circle_position, dc.CircleType.COUNTER_CLOCKWISE)
        end_cw_circle = dc.DubinCircle(end_cw_circle_position, dc.CircleType.CLOCKWISE)
        circles = [start_ccw_circle, start_cw_circle, end_ccw_circle, end_cw_circle]

        lengths = []
        paths = []

        for i in range(0, 2):
            for j in range(2, 4):
                path = self._calculate_two_circles_path(circles[i], circles[j])
                lengths.append(self._calculate_path_length(path))
                paths.append(path)


    def _find_CCC_solution(self):
        pass

    """
    Compute path between two Dubin circles,
        return total path length and a path.
    """
    def _calculate_two_circles_path(self, start_circle, end_circle):
        vec_start_end = end_circle.position - start_circle.position
        distance = np.linalg.norm(vec_start_end)
        vec_normalzied = vec_start_end / distance

        # When two circles are of different type and they are overlapping, no solution can be found.
        if distance < 2 * self.turning_radius and not start_circle.circle_type == end_circle.circle_type:
            return math.inf, []

        # Calculate the line segment that is tangent to both circles.

        # Angle difference between the vector from start circle to end circle
        # and the vector from the circle center to the tangent point.
        dtheta_start = 0
        dtheta_end = 0
        # When connecting two circles of same type.
        if start_circle.circle_type == end_circle.circle_type:
            if start_circle.circle_type == dc.CircleType.CLOCKWISE:
                dtheta_start = np.pi/2
            else: # counter clock-wise
                dtheta_start = -np.pi/2
            dtheta_end = dtheta_start

        # When connecting two circles of different type.
        else:
            if start_circle.circle_type == dc.CircleType.CLOCKWISE:
                dtheta_start = np.arccos(2 * self.turning_radius / distance)
            else:
                dtheta_start = -np.arccos(2 * self.turning_radius / distance)
            dtheta_end = np.pi + dtheta_start

        # Vector from center of the start circle to its tangent point.
        vec_tangent_point_start = np.array([[np.cos(dtheta_start), -np.sin(dtheta_start)],
                                            [np.sin(dtheta_start), np.cos(dtheta_start)]]) \
                                  @ vec_normalzied * self.turning_radius
        # Vector from center of the end circle to its tangent point.
        vec_tangent_point_end = np.array([[np.cos(dtheta_end), -np.sin(dtheta_end)],
                                            [np.sin(dtheta_end), np.cos(dtheta_end)]]) \
                                  @ vec_normalzied * self.turning_radius
        # Line segment that connects two tangent points on the two circles.
        segment_start = start_circle.position + vec_tangent_point_start
        segment_end = end_circle.position + vec_tangent_point_end

        # Construct a path
        path = []
        # The angle between the global x axis and the vector from circle center to path start position.
        start_circle_start_theta = self.start[2]
        # The angle between the global x axis and the vector from the circle center to its tangent point.
        start_circle_end_theta = start_circle_start_theta + dtheta_start
        path.append(CircularPath(start_circle.position,
                                 start_circle_start_theta,
                                 start_circle_end_theta,
                                 start_circle.circle_type,
                                 self.turning_radius))
        path.append(StraightPath(segment_start, segment_end))

        end_circle_start_theta = self.end[2]
        end_circle_end_theta = end_circle_start_theta + dtheta_end
        path.append(CircularPath(end_circle.position,
                                 end_circle_start_theta,
                                 end_circle_end_theta,
                                 end_circle.circle_type,
                                 self.turning_radius))

        return path

    def _calculate_path_length(self, path):

