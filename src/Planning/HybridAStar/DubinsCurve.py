import numpy as np
import src.Planning.HybridAStar.DubinCircle as dc
import math
from functools import reduce

class CircularPath:
    def __init__(self, position, start_theta, end_theta, circle_type, radius):
        self.position = position
        self.start_theta = (start_theta + 2 * np.pi) % (2 * np.pi)
        self.end_theta = (end_theta + 2 * np.pi) % (2 * np.pi)
        self.cirle_type = circle_type
        self.raidus = radius
    def length(self):
        # The angular distance from starting position to end position.
        path_theta = 0
        if self.cirle_type == dc.CircleType.COUNTER_CLOCKWISE:
            theta_end_start = self.end_theta - self.start_theta
            if theta_end_start >= 0:
                path_theta = theta_end_start
            else:
                path_theta = 2 * np.pi + theta_end_start
        # Clockwise path
        else:
            theta_start_end = self.start_theta - self.end_theta
            if theta_start_end >= 0:
                path_theta = theta_start_end
            else:
                path_theta = 2 * np.pi + theta_start_end
        return path_theta * self.raidus

class StraightPath:
    def __init__(self, start_position, end_position):
        self.start_position = start_position
        self.end_position = end_position
    def length(self):
        return np.linalg.norm(self.end_position - self.start_position)

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
        csc_path, csc_length = self._find_CSC_solution()
        # find CCC solution
        ccc_path, ccc_length = self._find_CCC_solution()

        if csc_length < ccc_length:
            return csc_path, csc_length
        else:
            return ccc_path, ccc_length

    def _find_CSC_solution(self):
        circles = self._find_tangent_cirlces()

        min_length = math.inf
        optimal_path = None
        for i in range(0, 2):
            for j in range(2, 4):
                path = self._calculate_two_circles_CSC_path(circles[i], circles[j])
                length = self._calculate_path_length(path)
                if length < min_length:
                    min_length = length
                    optimal_path = path

        return optimal_path, min_length

    def _find_CCC_solution(self):
        circles = self._find_tangent_cirlces()
        path_1 = self._calculate_two_circles_CCC_path(circles[0], circles[2])
        path_2 = self._calculate_two_circles_CCC_path(circles[1], circles[3])
        length_1 = self._calculate_path_length(path_1)
        length_2 = self._calculate_path_length(path_2)

        return path_1, length_1 \
            if length_1 < length_2 \
            else path_2, length_2

    """ Find path between two Dubin circle, path consists for three arcs, second arc has angle larger than pi. """
    def _calculate_two_circles_CCC_path(self, start_circle, end_circle):
        path = []
        # Tangent circle can only be found if both circles are of the same type.
        if start_circle.circle_type != end_circle.circle_type:
            print("Invalid circles")
            return []

        vec_start_end = end_circle.position - start_circle.position
        distance = np.linalg.norm(vec_start_end)
        vec_normalized = vec_start_end / distance

        # A tange circle between start and end circle could not be found
        # because they are too far away.
        if distance >= 4 * self.turning_radius:
            return []

        # Angle from vec_normalized to the tangent circle.
        dtheta_start = np.arccos(distance / (2 * self.turning_radius))
        # The tangent circle is on the right side of the vec_normalized
        if start_circle.circle_type == dc.CircleType.CLOCKWISE:
            dtheta_start = -dtheta_start
        dtheta_end = np.pi - dtheta_start

        # Vector from start circle to the tangent point.
        vec_tangent_point_start = np.array([[np.cos(dtheta_start), -np.sin(dtheta_start)],
                                              [np.sin(dtheta_start), np.cos(dtheta_start)]]) \
                                  @ vec_normalized * self.turning_radius
        # Vector from end circle to the tangent point.
        vec_tangent_point_end = np.array([[np.cos(dtheta_start), -np.sin(dtheta_start)],
                                          [np.sin(dtheta_start), np.cos(dtheta_start)]]) \
                                @ vec_normalized * self.turning_radius

        # Position of the two tangent points and tangent circle position.
        tangent_point_start = start_circle.position + vec_tangent_point_start
        tangent_point_end = end_circle.position + vec_tangent_point_end
        tangent_circle_position = start_circle.position + 2 * vec_tangent_point_start

        x_axis = np.array([1.0, 0])
        # Vector from center of the start circle center to the starting position of the path.
        vec_start_circle_start_position = self.start[0:2] - start_circle.position
        start_circle_start_theta = self._angle_diff(vec_start_circle_start_position, x_axis)
        start_circle_end_theta = self._angle_diff(vec_tangent_point_start, x_axis)
        path.append(CircularPath(start_circle.position,
                                 start_circle_start_theta,
                                 start_circle_end_theta,
                                 start_circle.circle_type,
                                 self.turning_radius))

        # Vector from tangent circle to the first tangent point
        vec_tangent_1 = tangent_point_start - tangent_circle_position
        vec_tangent_2 = tangent_point_end -  tangent_circle_position
        tangent_circle_start_theta = self._angle_diff(vec_tangent_1, x_axis)
        tangent_circle_end_theta = self._angle_diff(vec_tangent_2, x_axis)
        tangent_circle_type = dc.CircleType.CLOCKWISE \
                              if start_circle.circle_type == dc.CircleType.COUNTER_CLOCKWISE\
                              else dc.CircleType.COUNTER_CLOCKWISE

        path.append(CircularPath(tangent_circle_position,
                                 tangent_circle_start_theta,
                                 tangent_circle_end_theta,
                                 tangent_circle_type,
                                 self.turning_radius))

        vec_end_circle_end_position = self.end[0:2] - end_circle.position
        end_circle_start_theta = self._angle_diff(vec_tangent_point_end, x_axis)
        end_circle_end_theta = self._angle_diff(vec_end_circle_end_position, x_axis)
        path.append(CircularPath(end_circle.position,
                                 end_circle_start_theta,
                                 end_circle_end_theta,
                                 end_circle.circle_type,
                                 self.turning_radius))

        return path


    """ 
        Find four circles tangent to the starting position and the ending position. 
        First circle: Tangent to start point in counter clockwise direction.
        Second circle: Tangent to start point in clockwise direction.
        Third circle: Tangent to end point in counter clockwise direction.
        Fourth circl: Tangent to end point in clockwise direction.
    """
    def _find_tangent_cirlces(self):
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

        return [start_ccw_circle, start_cw_circle, end_ccw_circle, end_cw_circle]

    """
    Compute path between two Dubin circles, the path consists of arc, straight line and arc.
        return total path length and a path.
    """
    def _calculate_two_circles_CSC_path(self, start_circle, end_circle):
        vec_start_end = end_circle.position - start_circle.position
        distance = np.linalg.norm(vec_start_end)
        vec_normalized = vec_start_end / distance

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
                                  @ vec_normalized * self.turning_radius
        # Vector from center of the end circle to its tangent point.
        vec_tangent_point_end = np.array([[np.cos(dtheta_end), -np.sin(dtheta_end)],
                                            [np.sin(dtheta_end), np.cos(dtheta_end)]]) \
                                  @ vec_normalized * self.turning_radius
        # Line segment that connects two tangent points on the two circles.
        segment_start = start_circle.position + vec_tangent_point_start
        segment_end = end_circle.position + vec_tangent_point_end

        # Construct a path
        path = []
        # Vector from center of the start circle center to the starting position of the path.
        vec_start_circle_start_position = self.start[0:2] - start_circle.position
        x_axis = np.array([1.0, 0])
        # The angle between the global x axis and the vector from circle center to path start position.
        start_arc_start_theta = self._angle_diff(vec_start_circle_start_position, x_axis)
        # The angle between the global x axis and the vector from the circle center to its tangent point.
        start_arc_end_theta = self._angle_diff(vec_tangent_point_start, x_axis)
        # Add first circular/arc path.
        path.append(CircularPath(start_circle.position,
                                 start_arc_start_theta,
                                 start_arc_end_theta,
                                 start_circle.circle_type,
                                 self.turning_radius))
        # Add second straight path.
        path.append(StraightPath(segment_start, segment_end))

        vec_end_circle_end_position = self.end[0:2] - end_circle.position
        end_arc_start_theta = self._angle_diff(vec_tangent_point_end, x_axis)
        end_arc_end_theta = self._angle_diff(vec_end_circle_end_position, x_axis)
        # Add third circular/arc path.
        path.append(CircularPath(end_circle.position,
                                 end_arc_start_theta,
                                 end_arc_end_theta,
                                 end_circle.circle_type,
                                 self.turning_radius))

        return path

    def _calculate_path_length(self, path):
        if len(path) == 0:
            return 0
        return reduce(lambda sum, ele : sum + ele.length(), path)

    def _angle_diff(self, a, b):
        theta = np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        cross = np.cross(a, b)[0]
        if cross > 0:
            theta = -theta
        return theta
