import particle as p
import numpy as np
import copy
import ParticleFilter as pf


class FastSLAM:
    def __init__(self, numParticles, particle):
        self.particles = []
        for i in range(numParticles):
            self.particles.append(copy.deepcopy(particle))

    def motionUpdate(self, command, deltaT):
        for i in range(len(self.particles)):
            self.particles[i].motionUpdate(command, deltaT)

    def measurementUpdate(self, measurements):
        for i in range(len(self.particles)):
            self.particles[i].measurementUpdate(measurements)

    def getParticles(self):
        return self.particles

    def getBestParticle(self):
        logWeight = float("inf")
        bestParticle = self.particles[0]

        for i in range(len((self.particles))):
            weight = self.particles[i].getWeight()
            if weight < logWeight:
                logWeight = weight
                bestParticle = self.particles[i]

        return bestParticle

    def resample(self):
        weights = map(lambda p: p.getWeight(), self.particles)
        self.particles = pf.ParticleFilter(self.particles, weights)
