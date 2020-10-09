import src.OptimalControl.CostToGo as ctg
import src.Models.AircraftDynamics as ad
import src.OptimalControl.ValueIteration as vi
from src.OptimalControl.CostFunctions import *
import src.Utils.plot as uplt
from src.Utils.OptimalControlTestUtility import *
import numpy as np
import matplotlib.pyplot as plt
import pickle


cruise_speed=20.0
computation_delta_t = 1.0
simulation_delta_t = 0.3

def save_cost_to_file(table, file_name):
    file = open(file_name, 'wb')
    pickle.dump(table, file)

def get_dynamics():
    return ad.AircraftDynamics(step_time=0.01, cruise_speed=cruise_speed, fast_update=True)

def compute_cost_to_go_and_save(save_location, configs, num_iteration):
    cost_to_go = ctg.CostToGo(configs)
    dynamics = get_dynamics()
    cost_function = get_circle_cost_function(circle_center=[5, 5], radius=30, direction=1, direction_weight=500,
                                             direction_tau=0.05, control_weight=0)

    value_iteration = vi.ValueIteration(dynamics, cost_to_go, get_control_set(), cost_function, computation_delta_t)
    value_iteration.value_iteration(num_iteration)

    save_cost_to_file(value_iteration.get_cost_to_go(), save_location+'cost_to_go')
    save_cost_to_file(value_iteration.get_policy(), save_location+'policy')


def load_cost_to_go_and_display(location, configs):
    cost_to_go = pickle.load(open(location+'cost_to_go', 'rb'))
    cost_to_go_table = cost_to_go.get_state_space_cost_table()

    policy = pickle.load(open(location+'policy', 'rb'))
    policy_table = policy.get_state_space_cost_table()

    start_state = np.array([-30, -30, 3.14])
    path, _ = apply_control_policy(policy, get_dynamics(), start_state, simulation_delta_t, 100)
    uplt.plotRobotPoses(path)
    plt.ylim((configs[1]['min'], configs[1]['max']))
    plt.xlim((configs[0]['min'], configs[0]['max']))
    plt.show()

    draw_cost_to_go_and_policy(cost_to_go_table, policy_table, configs)


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

