from matplotlib import pyplot as plt
from common import get_example_splines, get_splines_x_y

if __name__ == '__main__':

    splines, arc_length_splines, seg_length = get_example_splines()

    X, Y = get_splines_x_y(splines, 1, step=50)
    X_al, Y_al = get_splines_x_y(arc_length_splines, seg_length, step=50)

    plt.plot(X, Y, 'b')
    plt.plot(X_al, Y_al, 'r')
    plt.xlim([0, 35])
    plt.ylim([0, 35])
    plt.title("Original in blue, reparameterized in red")
    plt.show()
