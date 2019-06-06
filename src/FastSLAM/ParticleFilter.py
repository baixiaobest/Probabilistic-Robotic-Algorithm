import numpy as np
import random

# Resample all the particles based on their current weights.
def ParticleFilter(particles, numParticles):
    normalizer = 0
    for i in range(numParticles):
        normalizer += particles[i][-1]

    for i in range(numParticles):
        particles[i][3] = particles[i][-1] / normalizer

    # Start resampling from old particles
    step = 1.0 / (numParticles)
    r = random.uniform(0.0, step)
    c = particles[0][-1]
    i = 0

    newParticles = []
    for m in range(numParticles):
        U = r + m * step
        while U > c:
            i = i + 1
            c = c + particles[i][-1]
        newParticle = np.array(particles[i], copy=True)
        newParticle[-1] = 1
        newParticles.append(newParticle)

    return newParticles