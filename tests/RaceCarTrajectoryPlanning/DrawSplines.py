from matplotlib import pyplot as plt
import numpy as np
import src.Planning.RaceCarTrajectoryPlanning.SplineFit as sf
import src.Planning.RaceCarTrajectoryPlanning.ArcLengthParameterizedSpline as alps

def get_splines_x_y(splines, end_t, step=50):
    t = np.linspace(0, end_t, step)
    X = np.array([])
    Y = np.array([])
    for spline in splines:
        a = spline[0]
        b = spline[1]
        c = spline[2]
        d = spline[3]
        x = a[0] * t ** 3 + b[0] * t ** 2 + c[0] * t + d[0]
        y = a[1] * t ** 3 + b[1] * t ** 2 + c[1] * t + d[1]
        X = np.concatenate((X, x))
        Y = np.concatenate((Y, y))

    return X, Y

if __name__ == '__main__':
    track_points = [
        np.array([5, 5]),
        np.array([5, 20]),
        np.array([10, 25]),
        np.array([15, 20]),
        np.array([20, 15]),
        np.array([25, 25]),
        np.array([30, 20])
    ]

    track_thetas = [
        np.pi / 2,
        np.pi / 2,
        0,
        -np.pi/2,
        np.pi/4,
        np.pi/4,
        -np.pi/2
    ]

    tangent_weights = [
        1, 10, 10, 10, 20, 10, 10
    ]

    track_tangents = []
    for i in range(len(track_thetas)):
        theta = track_thetas[i]
        weight = tangent_weights[i]
        track_tangents.append(np.array([np.cos(theta), np.sin(theta)]) * weight)

    splines=[]
    for i in range(len(track_points)-1):
        splines.append(
            sf.EndpointsSplineFit(track_points[i], track_points[i+1], track_tangents[i], track_tangents[i+1], end_t=1))

    X, Y = get_splines_x_y(splines, 1, step=50)

    processor = alps.ArcLengthParameterizedSplines()

    # Re-parameterize the spline with arc length.
    for spline in splines:
        processor.add_spline(spline)

    arc_length_splines, seg_length, _, _ = processor.compute_arc_length_parameterized_spline(10)
    X_al, Y_al = get_splines_x_y(arc_length_splines, seg_length, step=50)

    plt.plot(X, Y, 'b')
    plt.plot(X_al, Y_al, 'r')
    plt.xlim([0, 35])
    plt.ylim([0, 35])
    plt.title("Original in blue, reparameterized in red")
    plt.show()
