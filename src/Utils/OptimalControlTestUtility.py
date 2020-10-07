import numpy as np
import matplotlib.pyplot as plt


def apply_control_policy(control_policy, dynamics, start_state, delta_t, num_steps):
    state = np.array(start_state, copy=True)
    path = [state]
    for i in range(num_steps):
        control = control_policy.get_cost(state)
        state = dynamics.update(state, control, delta_t)
        path.append(state)

    return path

def draw_cost_to_go_and_policy(cost_to_go_table, policy_table, configs):
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