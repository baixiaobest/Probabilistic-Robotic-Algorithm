import numpy as np
import random
import copy

# Resample all the particles based on their current weights.
def ParticleFilter(particles, weights):
    normalizer = 0
    numParticles = len(particles)
    normalizedWeights = copy.deepcopy(weights)

    for i in range(numParticles):
        normalizer += weights[i]

    for i in range(numParticles):
        normalizedWeights[i] = normalizedWeights[i] / normalizer

    # Start resampling from old particles
    step = 1.0 / (numParticles)
    r = random.uniform(0.0, step)
    c = normalizedWeights[0]
    i = 0

    newParticles = []
    for m in range(numParticles):
        U = r + m * step
        while U > c:
            i = i + 1
            c = c + normalizedWeights[i]
        newParticle = copy.deepcopy(particles[i])
        newParticle.setWeight(1)
        newParticles.append(newParticle)

    return newParticles