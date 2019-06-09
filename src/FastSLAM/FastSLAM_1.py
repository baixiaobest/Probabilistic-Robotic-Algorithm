import particle as p
import numpy as np


class FastSLAM_1:
    def __init__(self, numParticles, initialPose, motionModel, measurementCovariance):
        self.particles = []
        for i in range(numParticles):
            self.particles.append(p.particle(initialPose, motionModel, measurementCovariance))

    def motionUpdate(self, command, deltaT):
        for i in range(len(self.particles)):
            self.particles[i].motionUpdate(command, deltaT)

    def measurementUpdate(self, measurements):
        for i in range(len(self.particles)):
            self.particles[i].measurementUpdate(measurements)

    def getParticles(self):
        return self.particles
