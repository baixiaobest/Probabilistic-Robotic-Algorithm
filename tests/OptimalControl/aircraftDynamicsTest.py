import unittest
import src.Models.AircraftDynamics as ad
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
        dynamics = ad.AircraftDynamics(step_time=0.001, cruise_speed=20.0, fast_update=False)
        start_state = np.array([10, 10, np.pi / 2.0])
        end_state = dynamics.update(start_state, -20, np.pi)
        correct_end_state = np.array([50, 10, 3.0/2.0 * np.pi])

        start_state2 = np.array([10, 10, np.pi / 2.0])
        end_state2 = dynamics.update(start_state2, 20, np.pi)
        correct_end_state2 = np.array([-30, 10, 3.0 / 2.0 * np.pi])

        start_state3 = np.array([10, 10, np.pi / 2.0])
        end_state3 = dynamics.update(start_state3, 20, np.pi / 2.0)
        correct_end_state3 = np.array([-10, 30, np.pi])

        start_state4 = np.array([10, 10, np.pi / 2.0])
        end_state4 = dynamics.update(start_state4, -20, np.pi / 2.0)
        correct_end_state4 = np.array([30, 30, 0])

        self.assertTrue(np.allclose(correct_end_state, end_state, atol=0.1))
        self.assertTrue(np.allclose(correct_end_state2, end_state2, atol=0.1))
        self.assertTrue(np.allclose(correct_end_state3, end_state3, atol=0.1))
        self.assertTrue(np.allclose(correct_end_state4, end_state4, atol=0.1))


def test_arc_fast(self):
        dynamics = ad.AircraftDynamics(step_time=0.001, cruise_speed=20.0, fast_update=True)
        start_state = np.array([10, 10, np.pi / 2.0])
        end_state = dynamics.update(start_state, -20, np.pi)
        correct_end_state = np.array([50, 10, 3.0/2.0 * np.pi])

        start_state2 = np.array([10, 10, np.pi / 2.0])
        end_state2 = dynamics.update(start_state2, 20, np.pi)
        correct_end_state2 = np.array([-30, 10, 3.0 / 2.0 * np.pi])

        start_state3 = np.array([10, 10, np.pi / 2.0])
        end_state3 = dynamics.update(start_state3, 20, np.pi/2.0)
        correct_end_state3 = np.array([-10, 30, np.pi])

        start_state4 = np.array([10, 10, np.pi / 2.0])
        end_state4 = dynamics.update(start_state4, -20, np.pi / 2.0)
        correct_end_state4 = np.array([30, 30, 0])

        self.assertTrue(np.allclose(correct_end_state, end_state, atol=0.1))
        self.assertTrue(np.allclose(correct_end_state2, end_state2, atol=0.1))
        self.assertTrue(np.allclose(correct_end_state3, end_state3, atol=0.1))
        self.assertTrue(np.allclose(correct_end_state4, end_state4, atol=0.1))


if __name__ == '__main__':
    unittest.main()
