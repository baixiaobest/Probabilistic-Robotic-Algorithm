import numpy as np
import math


class VelocityMotionModel:
    def __init__(self, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6):
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.alpha3 = float(alpha3)
        self.alpha4 = float(alpha4)
        self.alpha5 = float(alpha5)
        self.alpha6 = float(alpha6)

    # Sample a new pose from previous pose and control ut.
    # ut: a vector of [[speed], [angular speed]].
    # pose: a vector of [[x], [y], [theta]] representing robot pose.
    # deltaT: for how long this command is executing.
    # Returns: a new robot pose.
    def sampleNewPose(self, ut, pose, deltaT):
        v_cmd = ut[0, 0]
        w_cmd = ut[1, 0]
        x = pose[0, 0]
        y = pose[1, 0]
        theta = pose[2, 0]
        w_tolerance = 0.00000001

        v_hat = v_cmd + np.random.normal(0, math.sqrt(self.alpha1 * v_cmd*v_cmd + self.alpha2 * w_cmd*w_cmd))
        w_hat = w_cmd + np.random.normal(0, math.sqrt(self.alpha3 * v_cmd*v_cmd + self.alpha4 * w_cmd*w_cmd))
        gama_hat = np.random.normal(0, math.sqrt(self.alpha5 * v_cmd*v_cmd + self.alpha6 * w_cmd*w_cmd))

        if math.fabs(w_hat) > w_tolerance:
            v_over_w_hat = v_hat/w_hat
            x_prime = x - v_over_w_hat * math.sin(theta) + v_over_w_hat * math.sin(theta + w_hat * deltaT)
            y_prime = y + v_over_w_hat * math.cos(theta) - v_over_w_hat * math.cos(theta + w_hat * deltaT)
        else:
            x_prime = x + v_hat * deltaT * math.cos(theta)
            y_prime = y + v_hat * deltaT * math.sin(theta)

        theta_prime = theta + w_hat * deltaT + gama_hat * deltaT


        return np.array([[x_prime], [y_prime], [theta_prime]])
