import unittest

from matplotlib import pyplot as plt
import src.Planning.RaceCarTrajectoryPlanning.RaceLineGeneration as rlg
from common import get_example_splines, get_splines_x_y
import numpy as np

class MyTestCase(unittest.TestCase):
    def assertSequenceAlmostEqual(self, seq1, seq2, decimal):
        self.assertEqual(len(seq1), len(seq2))
        for i in range(len(seq1)):
            self.assertAlmostEqual(seq1[i], seq2[i], decimal)

    def assertMatrixAlmostEqual(self, M1, M2, decimal):
        h, w = M1.shape
        h2, w2 = M2.shape
        self.assertEqual(h, h2)
        self.assertEqual(w, w2)
        for row in range(h):
            for col in range(w):
                self.assertAlmostEqual(M1[row, col], M2[row, col], decimal)

    def test_dynamics(self):
        _, arc_length_splines, seg_length = get_example_splines()
        generator = rlg.RaceLineGeneration(arc_length_splines, seg_length, 0.1, 10, 3)
        # [x, y, heading, velocity, steering angle]
        states = np.array([10, 0, np.pi/2, 5, 0.2915])
        for i in range(100):
            states = generator._update_vehicle_nonlinear(states, np.zeros(2))
            x = states[0]
            y = states[1]
            distance = np.sqrt(x**2 + y**2)
            self.assertAlmostEqual(distance, 10, 2)

        states = np.array([-20, 0, np.pi / 2, 5, -0.14889])
        for i in range(100):
            states = generator._update_vehicle_nonlinear(states, np.zeros(2))
            x = states[0]
            y = states[1]
            distance = np.sqrt(x ** 2 + y ** 2)
            self.assertAlmostEqual(distance, 20, 2)

    def test_spline_position_tangent_angle(self):
        _, arc_length_splines, seg_length = get_example_splines()
        generator = rlg.RaceLineGeneration(arc_length_splines, seg_length, 0.1, 10, 3)
        length1 = 10
        pos1, theta1 = generator._splines_position_tangent_angle(length1)
        correct_position1 = [5, 15]
        correct_theta1 = np.pi/2
        self.assertSequenceAlmostEqual(pos1.tolist(), correct_position1, 2)
        self.assertAlmostEqual(theta1, correct_theta1, 2)

        length2 = 46.7388
        pos2, theta2 = generator._splines_position_tangent_angle(length2)
        correct_position2 = [22.6125, 18.7107]
        correct_theta2 = 1.1542
        self.assertSequenceAlmostEqual(pos2.tolist(), correct_position2, 2)
        self.assertAlmostEqual(theta2, correct_theta2, 2)

    def test_constraints(self):
        _, arc_length_splines, seg_length = get_example_splines()
        delta_t = 0.1
        N = 3
        generator = rlg.RaceLineGeneration(arc_length_splines, seg_length, delta_t=delta_t, horizon=N-1, L=3)
        prog_proj_vel_constr = generator._get_progress_projected_velocity_constraint()
        A1 = prog_proj_vel_constr.A[:, 5*N + 2*(N-1):]
        A1_correct = np.array([
            [1, -1,  0, delta_t,       0],
            [0,  1, -1,       0, delta_t]])
        self.assertMatrixAlmostEqual(A1, A1_correct, 2)

        # Visually inspect
        state_bound_constr = generator._get_state_bound_constraint()
        control_bound_constr = generator._get_control_bound_constraints()
        progress_bound_constr = generator._get_progress_bound_constraint()
        proj_vel_bound_constr = generator._get_projected_velocity_bound_constraint()

    def test_generate_race_line(self):
        _, arc_length_splines, seg_length = get_example_splines()
        generator = rlg.RaceLineGeneration(arc_length_splines, seg_length, delta_t=0.2, horizon=20, L=3)
        initial_states = np.array([6, 5, np.pi/2, 1, 0])
        generator.set_vehicle_states(initial_states)
        states, controls, progress, proj_vel = generator.generate_racing_line()
        traj_x = states[0, :]
        traj_y = states[1, :]
        traj_heading = states[2, :]
        traj_vel = states[3, :]
        traj_steer = states[4, :]

        X, Y = get_splines_x_y(arc_length_splines, seg_length, step=50)

        plt.plot(X, Y, 'b')
        plt.plot(traj_x, traj_y, 'r')
        plt.xlim([0, 35])
        plt.ylim([0, 35])
        plt.title("Original in blue, Trajectory in red")

        fig, ax = plt.subplots(5, 1)

        ax[0].plot(traj_vel)
        ax[0].set_title("velocity")

        ax[1].plot(np.rad2deg(traj_heading) % 360)
        ax[1].set_title("heading")

        ax[2].plot(np.rad2deg(traj_steer) % 360)
        ax[2].set_title("steering")

        ax[3].plot(progress)
        ax[3].set_title("progress")

        ax[4].plot(proj_vel)
        ax[4].set_title("projected velocity")

        fig, ax = plt.subplots(2, 1)

        ax[0].plot(controls[0, :])
        ax[0].set_title("acceleration")

        ax[1].plot(controls[1, :])
        ax[1].set_title("steer rate")

        plt.show()

if __name__ == '__main__':
    unittest.main()
