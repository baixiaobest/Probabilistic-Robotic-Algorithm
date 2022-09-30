import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.integrate import solve_ivp

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
        self.curr_progress = 0 # Vehicle current position projected to the spline.
        self.L = L
        self.num_states = 5
        self.num_cntrl = 2
        self.state_vec_size = self.num_states * self.N
        self.cntrl_vec_size = self.num_cntrl * (self.N - 1)
        self.progress_vec_size = self.N
        self.proj_vel_vec_size = self.N - 1 # Can be considered as rate of change of state of progress
        self.total_size = self.state_vec_size + \
                          self.cntrl_vec_size + \
                          self.progress_vec_size + \
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

        # Constraints
        self.vel_constr = [-10, 10]
        self.steer_constr = [-np.pi/3, np.pi/3]
        self.accel_constr = [-3, 3]
        self.steer_rate_constr = [-0.3, 0.3]

        self.constr_count = 0
        self.iter_count = 0

    def set_weights(self, w_cerr, w_lerr, w_progress, w_cntrl, w_proj_v):
        '''
        :param w_cerr: Weight for contouring error.
        :param w_lerr: Weight for lag error.
        :param w_progress: Weight for progress, or distance traveled at each time step projected to the splines.
        :param w_cntrl: Weight for control changes, shape of (2, 2)
        :param w_proj_v: Weight for change in projected velocity
        :return: None
        '''
        if w_cerr.shape != (self.num_cntrl, self.num_cntrl):
            raise ValueError(f"w_cntrl should be of size ({self.num_cntrl}, {self.num_cntrl})")
        self.w_cerr = w_cerr
        self.w_lerr = w_lerr
        self.w_progress = w_progress
        self.w_cntrl = w_cntrl
        self.w_proj_v = w_proj_v

    def set_vehicle_states(self, curr_states):
        self.curr_states = curr_states

    # def _project_position_to_splines(self, pos):
    #     '''
    #     Given position, find position on spline that is closest to it.
    #     :param pos: Position given.
    #     :return: Arc length starting from the beginning of splines.
    #     '''
    #     def objective(l):
    #

    def _update_vehicle_nonlinear(self, init_states, u):
        '''
        Update vehicle state by delta time.
        :param states: Vehicle states, numpy array of [x, y, heading, velocity, steering angle]
        :param u: Vehicle control, numpy array of [vehicle acceleration, steering angle rate]
        :return: Next vehicle states
        '''
        def f(t, states):
            theta = states[2]
            v = states[3]
            delta = states[4]

            dy_dt = np.zeros(5)
            dy_dt[0] = v * np.cos(theta)
            dy_dt[1] = v * np.sin(theta)
            dy_dt[2] = v * np.tan(delta) / self.L
            dy_dt[3] = u[0]
            dy_dt[4] = u[1]

            return dy_dt

        res = solve_ivp(f, (0, self.delta_t), init_states)

        return res.y[:, -1]

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
        '''
        Calculate the objective funciton.
        :param var: Design variables in the optimization problem.
        :return: Objective value.
        '''
        sum_cost_contour_err = 0
        sum_cost_lag_err = 0
        sum_progress_rewards = 0
        sum_proj_vel_change_cost = 0
        sum_control_cost = 0
        for i in range(self.N):
            # No computation of contouring error and lag error for the first state.
            if i > 1:
                # Calculate contouring error and lag error.
                x = var[self.num_states * i]
                y = var[self.num_states * i + 1]
                # State of progress, or length travel on spline
                l = var[self.state_vec_size + self.cntrl_vec_size + i]
                pos, theta = self._splines_position_tangent_angle(l)
                dx = x - pos[0]
                dy = y - pos[1]
                contour_err = ( np.sin(theta) * dx + np.cos(theta) * dy )**2
                lag_err = ( np.cos(theta) * dx - np.sin(theta) * dy )**2
                sum_cost_contour_err += contour_err
                sum_cost_lag_err += lag_err

            if i <= self.N-2:
                v = var[self.state_vec_size + self.cntrl_vec_size + self.progress_vec_size + i]
                sum_progress_rewards += v * self.delta_t

            if i <= self.N - 3:
                # Calculate reward for progress.
                v1 = var[self.state_vec_size + self.cntrl_vec_size + self.progress_vec_size + i]
                v2 = var[self.state_vec_size + self.cntrl_vec_size + self.progress_vec_size + i + 1]

                # Calculate cost of change in progress rate/projected velocity.
                dv = v2-v1
                sum_proj_vel_change_cost += dv**2

                # Calculate cost of control and cost of change in rate of progress.
                u1 = var[self.state_vec_size + self.num_cntrl * i : self.state_vec_size + self.num_cntrl * (i+1)]
                u2 = var[self.state_vec_size + self.num_cntrl * (i+1) : self.state_vec_size + self.num_cntrl * (i+2)]
                du = u2-u1
                sum_control_cost += du @ self.w_cntrl @ du

        val = sum_cost_contour_err * self.w_cerr \
                + sum_cost_lag_err * self.w_lerr \
                - sum_progress_rewards * self.w_progress \
                + sum_proj_vel_change_cost * self.w_proj_v \
                + sum_control_cost

        return val

    def _get_dynamics_constraint(self):
        '''
        Constraint due to vehicle dynamics.
        :return: Non-linear constraint.
        '''
        def constraint_dynamics_func(var):
            X_next = var[self.num_states:self.state_vec_size]
            X_curr = var[0:self.state_vec_size - self.num_states]
            U = var[self.state_vec_size: self.state_vec_size + self.cntrl_vec_size]
            f_x_u = np.zeros(self.state_vec_size - self.num_states)

            for i in range(self.N - 1):
                f_x_u[self.num_states * i: self.num_states * (i + 1)] = \
                    self._update_vehicle_nonlinear(
                        X_curr[self.num_states * i: self.num_states * (i + 1)],
                        U[self.num_cntrl * i: self.num_cntrl * (i + 1)])
            x_diff = X_next - f_x_u

            return x_diff

        return NonlinearConstraint(
            constraint_dynamics_func,
            -0.001*np.ones(self.num_states * (self.N - 1)),
            0.001*np.ones(self.num_states * (self.N - 1)))

    def _get_progress_projected_velocity_constraint(self):
        '''
        A matrix add linear constraint of
            p[k+1] = p[k] + delta_t * v[k],
            Where p[k] is progress at time step k, v[k] is projected velocity at time step k.
        :return: linear constraint
        '''
        A = np.zeros((self.N - 1, self.total_size))
        progress_location = self.state_vec_size + self.cntrl_vec_size
        A[0:self.N - 1, progress_location: progress_location + self.N - 1] += np.identity(self.N - 1)
        A[0:self.N - 1, progress_location + 1: progress_location + self.N] -= np.identity(self.N - 1)
        proj_vel_location = self.state_vec_size + self.cntrl_vec_size + self.progress_vec_size
        A[0:self.N - 1, proj_vel_location: proj_vel_location + self.N - 1] = self.delta_t * np.identity(self.N - 1)

        return LinearConstraint(A, np.zeros(self.N - 1), np.zeros(self.N - 1))

    def _get_state_bound_constraint(self):
        '''
        Constraints on state velocity and steering angle.
        :return: linear constraint
        '''
        cnstr_s = 2  # Number of constrained states
        A = np.zeros((cnstr_s * self.N, self.total_size))
        lb = np.zeros(cnstr_s * self.N)
        ub = np.zeros(cnstr_s * self.N)
        for i in range(self.N):
            A[cnstr_s * i: cnstr_s * (i + 1), self.num_states * i : self.num_states * (i + 1)] = \
                np.array([[0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]])
            lb[cnstr_s * i: cnstr_s * (i + 1)] = np.array([self.vel_constr[0], self.steer_constr[0]])
            ub[cnstr_s * i: cnstr_s * (i + 1)] = np.array([self.vel_constr[1], self.steer_constr[1]])

        return LinearConstraint(A, lb, ub)

    def _get_control_bound_constraints(self):
        '''
        Constraints on vehicle acceleration and steering rate.
        :return: linear constraint
        '''
        A = np.zeros((self.num_cntrl * (self.N-1), self.total_size))
        A[0:self.num_cntrl * (self.N-1), self.state_vec_size: self.state_vec_size + self.cntrl_vec_size] = \
            np.identity(self.num_cntrl * (self.N - 1))
        lb = np.zeros(self.num_cntrl * (self.N - 1))
        ub = np.zeros(self.num_cntrl * (self.N - 1))
        for i in range(self.N - 1):
            lb[self.num_cntrl * i: self.num_cntrl * (i + 1)] = np.array([self.accel_constr[0], self.steer_rate_constr[0]])
            ub[self.num_cntrl * i: self.num_cntrl * (i + 1)] = np.array([self.accel_constr[1], self.steer_rate_constr[1]])

        return LinearConstraint(A, lb, ub)

    def _get_progress_bound_constraint(self):
        '''
        Bounding on the state of progress.
        :return: Linear constraint.
        '''
        track_length = self.spline_length * len(self.splines)
        A = np.zeros((self.N, self.total_size))
        progress_location = self.state_vec_size + self.cntrl_vec_size
        A[0:self.N, progress_location : progress_location + self.N] = np.identity(self.N)
        lb = np.zeros(self.N)
        ub = track_length * np.ones(self.N)

        return LinearConstraint(A, lb, ub)

    def _get_projected_velocity_bound_constraint(self):
        '''
        Bound on projected velocity.
        :return: Linear constraint
        '''
        A = np.zeros((self.N - 1, self.total_size))
        proj_vel_location = self.state_vec_size + self.cntrl_vec_size + self.progress_vec_size
        A[0: self.N - 1, proj_vel_location: proj_vel_location + self.N - 1] = np.identity(self.N - 1)
        lb = self.vel_constr[0] * np.ones(self.N - 1)
        ub = self.vel_constr[1] * np.ones(self.N - 1)

        return LinearConstraint(A, lb, ub)

    def generate_racing_line(self):
        '''
        Generate racing lines and the controls required to execute the racing line.
        :return: list of states: numpy array of shape (number of states, horizon + 1)
                 list of controls: numpy array of shape (number of states, horizon)
        '''
        # Constraint first state.
        A1 = np.zeros((5, self.total_size))
        A1[0:5, 0:5] = np.identity(5)
        start_state_cnstr = LinearConstraint(A1, lb=self.curr_states, ub=self.curr_states)

        # TODO: Constraint the first state of progress.

        # Constraint due to dynamics.
        dynamics_constraint = self._get_dynamics_constraint()

        # State of Progress constraint
        progress_proj_vel_constraint = self._get_progress_projected_velocity_constraint()

        # TODO: Constraint on track boundary.

        # State constraints.
        state_constraint = self._get_state_bound_constraint()

        # Control constraints
        control_constraint = self._get_control_bound_constraints()

        # Bounding constraint on state of progress
        progress_constraint = self._get_progress_bound_constraint()

        # Bounding constraint on projected velocity
        projected_velocity_constraint = self._get_projected_velocity_bound_constraint()

        var = np.zeros(self.total_size)
        for i in range(self.N):
            var[self.num_states * i: self.num_states * (i + 1)] = self.curr_states

        def callback(xk, states):
            self.iter_count += 1
            print(f"iteration count: {self.iter_count} objective: {states.fun}")

        res = minimize(
            self._nonlinear_objective,
            var,
            method='SLSQP',
            constraints=[
                dynamics_constraint,
                progress_proj_vel_constraint,
                state_constraint,
                control_constraint,
                progress_constraint,
                projected_velocity_constraint
            ],
            options={"maxiter": 10, "disp": True},
            # callback=callback,
            tol=0.1)

        states_list = np.transpose(np.reshape(
                    res.x[0:self.state_vec_size],
                    (int(self.state_vec_size/self.num_states), self.num_states)))

        control_list = np.reshape(
            res.x[self.state_vec_size: self.state_vec_size + self.cntrl_vec_size],
            (int(self.cntrl_vec_size/self.num_cntrl), self.num_cntrl))

        progress = res.x[self.state_vec_size + self.cntrl_vec_size:
                         self.state_vec_size + self.cntrl_vec_size + self.progress_vec_size]

        proj_velocity = res.x[self.state_vec_size + self.cntrl_vec_size + self.progress_vec_size : ]

        return states_list, control_list, progress, proj_velocity

