import numpy as np

'''
Generate a trajectory of vehicle given reference trajectory defined by a list of splines.
Method is described by "Optimization-Based Autonomous Racing of 1:43 Scale RC Cars"
'''
class RaceLineGeneration:
    def __init__(self, splines, spline_length, delta_t, horizon, L):
        '''
        :param splines: A list of equal-length arc-length parameterized splines, list of spline parameters [[a, b, c, d], ...].
            Each spline is defined as x(l) = a*l^3 + b*l^2 + c*l + d, where l is [0, spline_length].
                                      y(l) = ...
            Where l is the arc length from the start of each spline.
        :param spline_length: Length of each spline.
        :param delta_t: Update time of the vehicle dynamics.
        :param horizon: Number of time steps into the future this class generates.
        :param L: Vehicle front wheel to rear wheel axis length.
        '''
        self.splines = splines
        self.spline_length = spline_length
        self.delta_t = delta_t
        self.N = horizon + 1 # N is number of states to be considered in the trajectory
        self.curr_states = np.zeros(5)
        self.L = L
        self.state_vec_size = 5 * self.N
        self.cntrl_vec_size = 2 * (self.N - 1)
        self.state_of_progress_vec_size = self.N
        self.proj_vel_vec_size = self.N - 1 # Can be considered as rate of change of state of progress
        self.total_size = self.state_vec_size + \
                          self.cntrl_vec_size + \
                          self.state_of_progress_vec_size + \
                          self.proj_vel_vec_size
        # Weight for contouring error.
        self.w_cerr = 1
        # Weight for lag error.
        self.w_lerr = 1
        # Weight for progress, or distance traveled at each time step projected to the splines.
        self.w_progress = 1
        # Weight for control changes.
        self.w_cntrl = np.identity(2)
        # Weight for change in projected velocity
        self.w_proj_v = 1

    def set_weights(self, w_cerr, w_lerr, w_progress, w_cntrl, w_proj_v):
        '''
        :param w_cerr: Weight for contouring error.
        :param w_lerr: Weight for lag error.
        :param w_progress: Weight for progress, or distance traveled at each time step projected to the splines.
        :param w_cntrl: Weight for control changes.
        :param w_proj_v: Weight for change in projected velocity
        :return: None
        '''
        self.w_cerr = w_cerr
        self.w_lerr = w_lerr
        self.w_progress = w_progress
        self.w_cntrl = w_cntrl
        self.w_proj_v = w_proj_v

    def set_vehicle_states(self, curr_states):
        self.curr_states = curr_states

    def _update_vehicle_nonlinear(self, states, u):
        '''
        Update vehicle state by delta time.
        :param states: Vehicle states, numpy array of [x, y, heading, velocity, steering angle]
        :param u: Vehicle control, numpy array of [vehicle acceleration, steering angle rate]
        :return: Next vehicle states
        '''
        x = states[0]
        y = states[1]
        theta = states[2]
        v = states[3]
        delta = states[4]

        new_states = np.zeros(5)
        new_states[0] = x + v * np.cos(theta) * self.delta_t
        new_states[1] = y + v * np.sine(theta) * self.delta_t
        new_states[2] = theta + v * np.tan(delta) / self.L
        new_states[3] = v + u[0] * self.delta_t
        new_states[4] = delta + u[1] * self.delta_t

    def _splines_position_tangent_angle(self, length):
        '''
        Starting from the beginning of the first spline,
        find the position after traversing the given length,
        as well as the angle between the x-axis and the tangent at this position.
        :param length: Length to traverse.
        :return: Position: position on the spline (numpy array vector).
                 Theta: Angle between x-axis and the tangent at this position.
        '''
        length = max(0, length)

        spline_idx = int(length / self.spline_length)
        l = length % self.spline_length

        # If length is beyond the total length of all the splines,
        # return the position and tangent at the end position of the splines.
        if spline_idx >= len(self.splines):
            spline_idx = len(self.splines) - 1
            l = self.spline_length

        spline = self.splines[spline_idx]
        a = spline[0]
        b = spline[1]
        c = spline[2]
        d = spline[3]

        pos = a*l**3 + b*l**2 + c*l + d
        tangent = 3*a*l**2 + 2*b*l + c
        theta = np.arctan2(tangent[1], tangent[0])

        return pos, theta

    def _nonlinear_objective(self, var):
        sum_cost_contour_err = 0
        sum_cost_lag_err = 0
        sum_progress_rewards = 0
        sum_proj_vel_change_cost = 0
        sum_control_cost = 0
        for i in range(len(self.N)):
            # No computation of contouring error and lag error for the first state.
            if i > 1:
                # Calculate contouring error and lag error.
                x = var[5*i]
                y = var[5*i + 1]
                # State of progress, or length travel on spline
                l = var[self.state_vec_size + self.cntrl_vec_size + i]
                pos, theta = self._splines_position_tangent_angle(l)
                dx = x - pos[0]
                dy = y - pos[1]
                contour_err = ( np.sin(theta) * dx + np.cos(theta) * dy )**2
                lag_err = ( np.cos(theta) * dx - np.sin(theta) * dy )**2
                sum_cost_contour_err += contour_err
                sum_cost_lag_err += lag_err

            if i < self.N:
                # Calculate reward for progress.
                v1 = var[self.state_vec_size + self.cntrl_vec_size + self.state_of_progress_vec_size + i]
                v2 = var[self.state_vec_size + self.cntrl_vec_size + self.state_of_progress_vec_size + i + 1]
                sum_progress_rewards += v1 * self.delta_t

                # Calculate cost of change in progress rate/projected velocity.
                dv = v2-v1
                sum_proj_vel_change_cost += dv**2

                # Calculate cost of control and cost of change in rate of progress.
                u1 = var[self.state_vec_size + 2*i : self.state_vec_size + 2*i + 2]
                u2 = var[self.state_vec_size + 2 * (i+1) : self.state_vec_size + 2 * (i+1) + 2]
                du = u2-u1
                sum_control_cost += du @ self.w_cntrl @ du

            return sum_cost_contour_err * self.w_cerr \
                   + sum_cost_lag_err * self.w_lerr \
                   - sum_progress_rewards * self.w_progress \
                   + sum_proj_vel_change_cost * self.w_proj_v \
                   + sum_control_cost * self.w_cntrl

    def generate_racing_line(self):
        '''
        Generate racing lines and the controls required to execute the racing line.
        :return: list of states, list of controls
        '''

        var = np.zeros(self.total_size)

