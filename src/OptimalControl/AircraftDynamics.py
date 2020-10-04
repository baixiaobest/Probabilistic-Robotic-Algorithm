import numpy as np

""" Simple lateral aircraft dynamics. """
class AircraftDynamics:
    """
    step_time: Time interval at each integration.
    cruise_speed: Aircraft cruise speed.
    """
    def __init__(self, step_time, cruise_speed):
        self.cruise_speed = cruise_speed
        self.step_time = step_time

    """ 
    Integrate the dynamics by delta_t amount of time.
        state: current state
        u: Lateral control acceleration m/s/s
        delta_t: total integration time
        return: new state after integration.
    """
    def update(self, state, u, delta_t):
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