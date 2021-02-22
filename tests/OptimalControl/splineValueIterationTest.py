import src.OptimalControl.DynamicProgramming.CostToGo as ctg
import src.Models.AircraftDynamics as ad
import src.OptimalControl.DynamicProgramming.ValueIteration as vi
from src.OptimalControl.DynamicProgramming.CostFunctions import *
from src.OptimalControl.DynamicProgramming.VectorCubicSpline import create_spline_start_end_point_velocity
import src.OptimalControl.DynamicProgramming.CachedSplineDistance as cs
import src.Utils.plot as uplt
from src.Utils.OptimalControlTestUtility import *
import numpy as np
import matplotlib.pyplot as plt
import pickle


cruise_speed=20.0
computation_delta_t = 0.5
simulation_delta_t = 0.3

def save_cost_to_file(table, file_name):
    file = open(file_name, 'wb')
    pickle.dump(table, file)

def get_dynamics():
    return ad.AircraftDynamics(step_time=0.01, cruise_speed=cruise_speed, fast_update=True)

def get_spline():
    return create_spline_start_end_point_velocity(start=[-60, 30], start_vel=[300, 0], end=[60, -30], end_vel=[300, 0])

def get_cost_function():
    spline = get_spline()
    cached_spline = cs.CachedSplineDistance(spline, configs[0:2])
    cached_spline.compute_cache()
    cost_function = get_spline_cost_function(cached_spline=cached_spline, direction_weight=100, direction_tau=0.05,
                                             control_weight=0)
    return cost_function

def compute_cost_to_go_and_save(save_location, configs, num_iteration):
    cost_to_go = ctg.CostToGo(configs)
    dynamics = get_dynamics()

    value_iteration = vi.ValueIteration(dynamics, cost_to_go, get_control_set(), get_cost_function(), computation_delta_t)
    value_iteration.value_iteration(num_iteration)

    save_cost_to_file(value_iteration.get_cost_to_go(), save_location+'spline_cost_to_go')
    save_cost_to_file(value_iteration.get_policy(), save_location+'spline_policy')


def load_cost_to_go_and_display(location, configs):
    cost_to_go = pickle.load(open(location+'spline_cost_to_go', 'rb'))
    cost_to_go_table = cost_to_go.get_state_space_cost_table()

    policy = pickle.load(open(location+'spline_policy', 'rb'))
    policy_table = policy.get_state_space_cost_table()

    # value_iteration = vi.ValueIteration(get_dynamics(), cost_to_go, get_control_set(), get_cost_function(), computation_delta_t)
    # value_iteration.compute_control_policy()
    # policy = value_iteration.get_policy()
    # policy_table = policy.get_state_space_cost_table()

    draw_spline(get_spline())

    start_state = np.array([-50, 30, 0])
    path, controls = apply_control_policy(policy, get_dynamics(), start_state, simulation_delta_t, 30)
    print(controls)
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
    configs = [{"min": -80, "max": 80, "resolution": 1},
               {"min": -80, "max": 80, "resolution": 1},
               {"min": 0, "max": 2 * np.pi, "resolution": np.pi/8.0}]
    num_iteration = 10

    if compute:
        compute_cost_to_go_and_save(save_location, configs, num_iteration)
    else:
        cost_to_go = load_cost_to_go_and_display(save_location, configs)

