import numpy as np

""" Simple lateral aircraft dynamics. """
class AircraftDynamics:
    """
    step_time: Time interval at each integration.
    cruise_speed: Aircraft cruise speed.
    """
    def __init__(self, step_time, cruise_speed, fast_update=False):
        self.cruise_speed = cruise_speed
        self.step_time = step_time
        self.fast_update = fast_update

    """ 
    Integrate the dynamics by delta_t amount of time.
        state: current state
        u: Lateral control acceleration m/s/s
        delta_t: total integration time
        return: new state after integration.
    """
    def update(self, state, u, delta_t):
        if self.fast_update:
            return self._fast_update(state, u, delta_t)
        else:
            return self._slow_update(state, u, delta_t)

    def _fast_update(self, state, u, delta_t):
        new_state = np.zeros(state.shape)
        theta = state[2]

        if np.abs(u) > 0.001:
            turn_radius = self.cruise_speed ** 2 / u
            vec_aircraft_circle_center = np.array([-np.sin(theta), np.cos(theta)]) * turn_radius
            vec_cir_centr_start_aircrft = -vec_aircraft_circle_center
            turn_center = state[0:2] + vec_aircraft_circle_center
            delta_theta = self.cruise_speed / turn_radius * delta_t
            vec_cir_centr_end_aircraft = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                                                   [np.sin(delta_theta), np.cos(delta_theta)]]) \
                                         @ vec_cir_centr_start_aircrft
            aircraft_end_pos = turn_center + vec_cir_centr_end_aircraft
            new_state[0] = aircraft_end_pos[0]
            new_state[1] = aircraft_end_pos[1]
            new_state[2] = (theta + delta_theta) % (2*np.pi)
        else:
            new_state[0] = state[0] + np.cos(theta) * self.cruise_speed * delta_t
            new_state[1] = state[1] + np.sin(theta) * self.cruise_speed * delta_t
            new_state[2] = theta

        return new_state


    def _slow_update(self, state, u, delta_t):
        new_state = np.array(state, copy=True)
        # Integrate integer number of steps.
        for i in range(int(delta_t / self.step_time)):
            new_state = self._integrate(new_state, u, self.step_time)

        # Integrate the rest of the time.
        if delta_t % self.step_time > 0:
            new_state = self._integrate(new_state, u, delta_t % self.step_time)

        return new_state
    """ Euler integration in delta_t time. """
    def _integrate(self, state, u, delta_t):
        new_state = np.zeros(state.shape)
        x = state[0]
        y = state[1]
        theta = state[2]
        new_state[0] = x + self.cruise_speed * np.cos(theta) * delta_t
        new_state[1] = y + self.cruise_speed * np.sin(theta) * delta_t
        new_state[2] = (theta + u / self.cruise_speed * delta_t) % (2 * np.pi)

        return new_state