
import pydrake
import underactuated
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from numpy.linalg import inv

from pydrake.all import (DiagramBuilder, LinearQuadraticRegulator,
                         PlanarSceneGraphVisualizer, Saturation,
                         SceneGraph, Simulator, WrapToSystem, SignalLogger, LeafSystem,
                         PortDataType, BasicVector, Linearize)
from pydrake.examples.acrobot import (AcrobotInput, AcrobotGeometry,
                                      AcrobotPlant, AcrobotState, AcrobotSpongController)
from underactuated.jupyter import AdvanceToAndVisualize, SetupMatplotlibBackend, running_as_notebook
plt_is_interactive = SetupMatplotlibBackend()

def UprightState():
    state = AcrobotState()
    state.set_theta1(np.pi)
    state.set_theta2(0.)
    state.set_theta1dot(0.)
    state.set_theta2dot(0.)
    return state


def BalancingLQR():
    # Design an LQR controller for stabilizing the Acrobot around the upright.
    # Returns a (static) AffineSystem that implements the controller (in
    # the original AcrobotState coordinates).

    acrobot = AcrobotPlant()
    context = acrobot.CreateDefaultContext()

    input = AcrobotInput()
    input.set_tau(0.)
    context.FixInputPort(0, input)

    context.get_mutable_continuous_state_vector()\
        .SetFromVector(UprightState().CopyToVector())

    Q = np.diag((10., 10., 1., 1.))
    R = [1]

    return LinearQuadraticRegulator(acrobot, context, Q, R)

class EnergyShapingControl(LeafSystem):
    def __init__(self, k1, k2, k3, lqr_angle_range=1.0):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.acrobot_plant = AcrobotPlant()
        self.acrobot_context = self.acrobot_plant.CreateDefaultContext()
        self.upright_context = self.acrobot_plant.CreateDefaultContext()
        self.upright_context.get_mutable_continuous_state()\
            .SetFromVector(np.array([np.pi, 0, 0, 0]))
        self.lqr_angle_range = lqr_angle_range

        self.DeclareInputPort('estimated state', PortDataType.kVectorValued, 4)
        self.DeclareVectorOutputPort('control', BasicVector(1), self.calculate_control)
        self.Q = np.diag((10., 10., 1., 1.))
        self.R = [1]
        self.K = self.calculate_lqr()
        self.LQR_on = False

    def calculate_lqr(self):
        upright_plant = AcrobotPlant()
        upright_plant_context = upright_plant.CreateDefaultContext()
        upright_plant_context.get_mutable_continuous_state() \
            .SetFromVector(np.array([np.pi, 0, 0, 0]))
        upright_plant.GetInputPort("elbow_torque") \
            .FixValue(upright_plant_context, 0)

        lqr = LinearQuadraticRegulator(upright_plant, upright_plant_context, self.Q, self.R)
        return lqr.D()

    def calculate_control(self, context, control_vec):
        input = self.get_input_port(0).Eval(context)
        theta1 = input[0]
        theta2 = input[1]
        theta1_dot = input[2]
        theta2_dot = input[3]

        # Current acrobot context based on input.
        self.acrobot_context \
            .get_mutable_continuous_state() \
            .SetFromVector(input)

        desired_state = np.array([np.pi, 0, 0, 0])
        cost = (desired_state - input).T @ self.Q @ (desired_state - input)
        should_use_lqr = cost < 50

        if not self.LQR_on:
            self.LQR_on = should_use_lqr
            if should_use_lqr:
                print("LQR on {0}".format(context.get_time()))

        if self.LQR_on:
            u = -self.K @ (desired_state - input)
            control_vec.SetFromVector(u)
        else:
            # Bias term of the dynamics.
            bias = self.acrobot_plant.DynamicsBiasTerm(self.acrobot_context)

            # Mass matrix
            M = self.acrobot_plant.MassMatrix(self.acrobot_context)
            m11 = M[0, 0]
            m12 = M[0, 1]
            m21 = M[1, 0]
            m22 = M[1, 1]
            m22_bar = m22 - m21 * m12 / m11
            bias_bar = bias[1] - m21 / m11 * bias[0]

            # Desired energy, upright configuration.
            Ed = self.acrobot_plant.EvalPotentialEnergy(self.upright_context) \
                 + self.acrobot_plant.EvalKineticEnergy(self.upright_context)
            # Current energy.
            Ec = self.acrobot_plant.EvalPotentialEnergy(self.acrobot_context) \
                 + self.acrobot_plant.EvalKineticEnergy(self.acrobot_context)
            E_error = Ec - Ed

            u_bar = E_error * theta1_dot

            u = -self.k1 * theta2 - self.k2 * theta2_dot + self.k3 * u_bar
            tau = m22_bar * u + bias_bar

            control_vec.SetFromVector(np.array([tau]))


if __name__=="__main__":

    builder = DiagramBuilder()
    acrobot = builder.AddSystem(AcrobotPlant())

    saturation = builder.AddSystem(Saturation(min_value=[-10], max_value=[10]))
    builder.Connect(saturation.get_output_port(0), acrobot.get_input_port(0))
    wrapangles = WrapToSystem(4)
    wrapangles.set_interval(0, 0, 2. * np.pi)
    wrapangles.set_interval(1, -np.pi, np.pi)
    wrapto = builder.AddSystem(wrapangles)
    builder.Connect(acrobot.get_output_port(0), wrapto.get_input_port(0))

    # controller = builder.AddSystem(BalancingLQR())
    # builder.Connect(wrapto.get_output_port(0), controller.get_input_port(0))
    
    controller = builder.AddSystem(EnergyShapingControl(k1=50, k2=5, k3=5.0))
    builder.Connect(wrapto.get_output_port(0), controller.get_input_port(0))

    builder.Connect(controller.get_output_port(0), saturation.get_input_port(0))

    logger = builder.AddSystem(SignalLogger(4))
    builder.Connect(acrobot.get_output_port(0), logger.get_input_port(0))
    u_logger = builder.AddSystem(SignalLogger(1))
    builder.Connect(controller.get_output_port(0), u_logger.get_input_port(0))

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    # Simulate
    duration = 20.0

    context.SetTime(0.)
    context.SetContinuousState(np.array([0.0, 0, 0.1, 0]))
    simulator.Initialize()
    simulator.AdvanceTo(duration)
    data = np.copy(logger.data())
    times = logger.sample_times()
    u_data = np.copy(u_logger.data())

    fig, axs = plt.subplots(5)

    for i in range(data.shape[1]):
        data[0, i] = (data[0, i]) % (2*np.pi)
        data[1, i] = (data[1, i]) % (2*np.pi)

    theta0_plt, = axs[0].plot(times, data[0, :])
    theta1_plt, = axs[1].plot(times, data[1, :])
    theta0_dot_plt, = axs[2].plot(times, data[2, :])
    theta1_dot_plt, = axs[3].plot(times, data[3, :])
    u_plt, = axs[4].plot(times, u_data[0, :])

    labels = ["Theta 0", "Theta 1", "Omega 0", "Omega 1", "u"]

    for i in range(len(labels)):
        axs[i].set(xlabel='time', ylabel=labels[i])

    plt.show()
