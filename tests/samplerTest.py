import src.Utils.sampler as s
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    N = 100000
    mu = 5
    sigma = 3
    arr = []
    for i in range(N):
        arr.append(s.sampleNormal(mu, sigma))
    plt.hist(arr, np.linspace(-5, 15, 1000))
    plt.show()
    print np.std(arr)