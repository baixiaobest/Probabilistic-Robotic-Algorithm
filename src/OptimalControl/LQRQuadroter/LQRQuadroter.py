import pydrake
import underactuated
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from numpy.linalg import inv

from pydrake.all import (DiagramBuilder, LinearQuadraticRegulator, Saturation,
                         SceneGraph, Simulator, WrapToSystem, SignalLogger, LeafSystem,
                         PortDataType, BasicVector, Linearize, InputPortIndex, OutputPortIndex)
from pydrake.examples.quadrotor import (QuadrotorPlant)

class QuadLQRController(LeafSystem):
    def __init__(self, Q=None, R=None):
        super().__init__()
        self.DeclareInputPort('quadrotor state', PortDataType.kVectorValued, 12)
        self.DeclareVectorOutputPort('control', BasicVector(4), self.calculate_control)

        self.m = 0.775
        self.L = 0.15
        self.kF = 1.0
        self.kM = 0.0245
        self.I = np.array([[0.0015, 0, 0],
                           [0, 0.0025, 0],
                           [0, 0, 0.0035]])
        self.g = 9.81
        self.nominal_state = [0,]*12
        if Q is None:
            self.Q = np.diag([10, 10, 10, 20, 20, 10, 10, 10, 10, 1, 1, 1])
        else:
            self.Q = Q
        if R is None:
            self.R = np.diag([1, 1, 1, 1])
        else:
            self.R = R
        (K, S) = self.get_LQR_params()
        self.K = K

    def get_LQR_params(self):
        quadroter = QuadrotorPlant()
        context = quadroter.CreateDefaultContext()
        hover_thrust = self.m * self.g
        quadroter.get_input_port(0).FixValue(context, [hover_thrust / 4.0,] * 4)
        context.get_mutable_continuous_state_vector()\
               .SetFromVector(self.nominal_state)
        linear_sys = Linearize(quadroter, context, InputPortIndex(0), OutputPortIndex(0))
        return LinearQuadraticRegulator(linear_sys.A(), linear_sys.B(), self.Q, self.R)

    def calculate_control(self, context, control_vec):
        input = self.get_input_port(0).Eval(context)
        u_bar = -self.K @ input
        u_0 = np.array([self.m * self.g / 4.0,] * 4)
        output = u_0 + u_bar
        control_vec.SetFromVector(output)


if __name__=="__main__":
    builder = DiagramBuilder()
    quadroter = builder.AddSystem(QuadrotorPlant())

    limit = 10
    saturation = builder.AddSystem(Saturation(min_value=[-limit,]*4, max_value=[limit,]*4))
    builder.Connect(saturation.get_output_port(0), quadroter.get_input_port(0))

    controller = builder.AddSystem(QuadLQRController())
    builder.Connect(quadroter.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), saturation.get_input_port(0))

    logger = builder.AddSystem(SignalLogger(12))
    builder.Connect(quadroter.get_output_port(0), logger.get_input_port(0))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    duration = 20.0

    context.SetTime(0.0)
    # x, y, z, r, p, y, vx, vy, vz, vr, vp, vy
    context.SetContinuousState(np.array([0, 0, 0, -3.14, 0, 0, 0, 0, 0, 0, 0, 0]))
    simulator.Initialize()
    simulator.AdvanceTo(duration)
    data = np.copy(logger.data())
    times = logger.sample_times()

    x = data[0, :]
    y = data[1, :]
    z = data[2, :]
    r = data[3, :]
    p = data[4, :]
    yaw = data[5, :]

    fig, axs = plt.subplots(3)
    x_plt, = axs[0].plot(times, x)
    y_plt, = axs[1].plot(times, y)
    z_plt, = axs[2].plot(times, z)
    label_1 = ["x", "y", "z"]

    for i in range(3):
        axs[i].set(xlabel="time", ylabel=label_1[i])

    fig, axs = plt.subplots(3)
    r_plt, = axs[0].plot(times, r)
    p_plt, = axs[1].plot(times, p)
    yaw_plt, = axs[2].plot(times, yaw)

    label_2 = ["r", "p", "y"]

    for i in range(3):
        axs[i].set(xlabel="time", ylabel=label_2[i])

    plt.show()