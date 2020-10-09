import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def apply_control_policy(control_policy, dynamics, start_state, delta_t, num_steps):
    state = np.array(start_state, copy=True)
    path = [state]
    controls = []
    for i in range(num_steps):
        try:
            control = control_policy.get_cost(state)
        except:
            break
        state = dynamics.update(state, control, delta_t)
        path.append(state)
        controls.append(control)

    return path, controls

def draw_cost_to_go_and_policy(cost_to_go_table, policy_table, configs):
    shape = cost_to_go_table.shape
    for i in range(cost_to_go_table.shape[2]):
        cost_table_slice = cost_to_go_table[:, :, i]
        policy_table_slice = policy_table[:, :, i]
        print(policy_table_slice)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')

        X = np.arange(configs[0]['min'], configs[0]['max'] + configs[0]['resolution'], configs[0]['resolution'])[:shape[0]]
        Y = np.arange(configs[1]['min'], configs[1]['max'] + configs[1]['resolution'], configs[1]['resolution'])[:shape[1]]
        X, Y = np.meshgrid(X, Y)

        theta = configs[2]['min'] + configs[2]['resolution'] * i

        surf = ax.plot_surface(X, Y, cost_table_slice, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_title(label="Cost (theta={0})".format(theta))

        surf2 = ax2.plot_surface(X, Y, policy_table_slice, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax2.set_title(label="Policy (theta={0})".format(theta))

        plt.show()

def draw_spline(spline, N=100):
    x_list = []
    y_list = []
    for i in range(N):
        s = i * 1.0 / N
        point = spline.get_point(s)
        x_list.append(point[0])
        y_list.append(point[1])

    plt.plot(x_list, y_list, 'ro-', markersize=1)