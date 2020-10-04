import unittest
import src.OptimalControl.AircraftDynamics as ad
import numpy as np

class AircraftDynamicsTest(unittest.TestCase):
    def test_stright_line(self):
        dynamics = ad.AircraftDynamics(step_time=0.001, cruise_speed=20.0)
        start_state = np.array([10, 10, np.pi/4.0])
        end_state = dynamics.update(start_state, 0, 10.0)
        dx = 200.0/np.sqrt(2)
        dy = dx
        correct_end_state = np.array([10+dx, 10+dy, np.pi/4.0])

        self.assertTrue(np.allclose(correct_end_state, end_state, atol=0.1))

    def test_arc(self):
        dynamics = ad.AircraftDynamics(step_time=0.001, cruise_speed=20.0)
        start_state = np.array([0, 0, np.pi / 2.0])
        end_state = dynamics.update(start_state, -20, np.pi)
        correct_end_state = np.array([40, 0, 3.0/2.0 * np.pi])

        self.assertTrue(np.allclose(correct_end_state, end_state, atol=0.1))


if __name__ == '__main__':
    unittest.main()
