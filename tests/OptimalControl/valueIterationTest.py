import src.OptimalControl.CostToGo as ctg
import src.OptimalControl.AircraftDynamics as ad
import src.OptimalControl.ValueIteration as vi
import src.Utils.plot as uplt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle


cruise_speed=20.0
delta_t = 0.3

def get_cost_function(target_x, target_y, control_weight, velocity_weight):
    def cost_function(state, control):
        x_dist_to_target = target_x - state[0]
        y_dist_to_target = target_y - state[1]
        velocity_to_target = cruise_speed * np.sin(state[2])
        return y_dist_to_target ** 2 + x_dist_to_target**2 + control_weight * control ** 2
    return cost_function


def save_cost_to_file(table, file_name):
    file = open(file_name, 'wb')
    pickle.dump(table, file)

def get_dynamics():
    return ad.AircraftDynamics(step_time=0.01, cruise_speed=cruise_speed, fast_update=True)

def compute_cost_to_go_and_save(save_location, configs, num_iteration):
    cost_to_go = ctg.CostToGo(configs)
    dynamics = get_dynamics()
    cost_function = get_cost_function(target_x=0, target_y=0, control_weight=0, velocity_weight=0)

    value_iteration = vi.ValueIteration(dynamics, cost_to_go, get_control_set(), cost_function, delta_t)
    value_iteration.value_iteration(num_iteration)

    save_cost_to_file(value_iteration.get_cost_to_go(), save_location+'cost_to_go')
    save_cost_to_file(value_iteration.get_policy(), save_location+'policy')


def load_cost_to_go_and_display(location, configs):
    cost_to_go = pickle.load(open(location+'cost_to_go', 'rb'))
    cost_to_go_table = cost_to_go.get_state_space_cost_table()

    policy = pickle.load(open(location+'policy', 'rb'))
    policy_table = policy.get_state_space_cost_table()

    start_state = np.array([10, 10, 1.57])
    path = apply_control_policy(policy, start_state, delta_t, 50)
    uplt.plotRobotPoses(path)
    plt.show()

    for i in range(cost_to_go_table.shape[2]):
        cost_table_slice = cost_to_go_table[:, :, i]
        policy_table_slice = policy_table[:, :, i]
        print(policy_table_slice)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')

        X = np.arange(configs[0]['min'], configs[0]['max'] + configs[0]['resolution'], configs[0]['resolution'])
        Y = np.arange(configs[1]['min'], configs[1]['max'] + configs[1]['resolution'], configs[1]['resolution'])
        X, Y = np.meshgrid(X, Y)

        theta = configs[2]['min'] + configs[2]['resolution'] * i

        surf = ax.plot_surface(X, Y, cost_table_slice, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_title(label="Cost (theta={0})".format(theta))

        surf2 = ax2.plot_surface(X, Y, policy_table_slice, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax2.set_title(label="Policy (theta={0})".format(theta))

        plt.show()

def apply_control_policy(control_policy, start_state, delta_t, num_steps):
    state = np.array(start_state, copy=True)
    dynamics = get_dynamics()
    path = [state]
    for i in range(num_steps):
        control = control_policy.get_cost(state)
        state = dynamics.update(state, control, delta_t)
        path.append(state)

    return path

def get_control_set():
    g = 9.8
    return np.linspace(-2 * g, 2 * g, 20)

if __name__=="__main__":
    compute = False

    save_location = '../../cache/'
    configs = [{"min": -60, "max": 60, "resolution": 2},
               {"min": -60, "max": 60, "resolution": 2},
               {"min": 0, "max": 2 * np.pi, "resolution": np.pi/8.0}]
    num_iteration = 5


    if compute:
        compute_cost_to_go_and_save(save_location, configs, num_iteration)
    else:
        cost_to_go = load_cost_to_go_and_display(save_location, configs)

