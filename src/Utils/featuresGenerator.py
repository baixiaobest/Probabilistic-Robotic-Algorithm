import random as r
import numpy as np

def generateFeatures(xmin, xmax, ymin, ymax, numFeatures):
    features = []
    for i in range(numFeatures):
        x = r.uniform(xmin, xmax)
        y = r.uniform(ymin, ymax)
        features.append(np.array([x, y]))
    return features