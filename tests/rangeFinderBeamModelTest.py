import matplotlib.pyplot as plt
import src.Models.rangeFinderBeamModel as rm
import numpy as np
import src.Utils.plot as plot


def probFunctionsTest():
    w_hit = 0.6
    w_short = 0.1
    w_rand = 0.1
    w_max = 0.2
    sigma_hit = 0.5
    lamda_short = 0.1
    z_max = 10
    expec_z = 5

    model = rm.RangeFinderBeamModel(w_hit, w_short, w_max, w_rand, sigma_hit, lamda_short, z_max)
    points = np.linspace(0, z_max + 1, 1000)
    phit = []
    pshort = []
    pmax = []
    prand = []
    ptotal = []
    for i in range(points.shape[0]):
        phit.append(model._calculatePhit(points[i], expec_z))
        pshort.append(model._calculatePshort(points[i], expec_z))
        pmax.append(model._calculatePmax(points[i]))
        prand.append(model._calculatePrand(points[i]))
        ptotal.append(model._calculateTotalProb(points[i], expec_z))

    plt.plot(points, phit)
    plt.plot(points, pshort)
    plt.plot(points, pmax)
    plt.plot(points, prand)
    plt.show()
    plt.plot(points, ptotal)
    plt.show()

def raycastTest():
    w_hit = 0.6
    w_short = 0.2
    w_rand = 0.1
    w_max = 0.1
    sigma_hit = 0.5
    lamda_short = 0.1
    z_max = 10

    model = rm.RangeFinderBeamModel(w_hit, w_short, w_max, w_rand, sigma_hit, lamda_short, z_max)
    grid = plot.readBMPAsNumpyArray("../map/maze_map.bmp")

    measurements = np.linspace(0, 20, 1000)
    pose = np.array([4, 12])
    prob = []

    for i in range(len(measurements)):
        zt = measurements[i]
        prob.append(model.calculateProb([np.array([0, zt])], pose, grid, resolution=0.01))

    plt.plot(measurements, prob)
    plt.show()


if __name__ == '__main__':
    raycastTest()
