import random
import math
import numpy as np


class MonteCarloLocalization:
    # grid: Occupancy grid.
    # poseGuess: Initial guess of robot position. [XMin, XMax, YMin, YMax].
    # numParticles: Number of particles used.
    # motionModel: Probabilistic motion model.
    # measurementModel: Probabilistic measurement model.
    def __init__(self, grid, resolution, poseGuess, numParticles, motionModel, measurementModel):
        self.grid = grid
        self.resolution = resolution
        self.numParticles = numParticles
        self.motionModel = motionModel
        self.measurementModel = measurementModel
        self.particles = []

        xMin = poseGuess[0]
        xMax = poseGuess[1]
        yMin = poseGuess[2]
        yMax = poseGuess[3]
        # Generate a set of particles.
        # Particle is represented as [x, y, theta, weight]
        for i in range(numParticles):
            x = random.uniform(xMin, xMax)
            y = random.uniform(yMin, yMax)
            theta = random.uniform(0, 2.0 * math.pi)
            self.particles.append(np.array([x, y, theta, 1]))

    # Sample a set of new particles from previous particles using motion model.
    # The sample set represents the proposal distribution.
    def motionUpdate(self, ut, deltaT):
        for i in range(self.numParticles):
            self.particles[i][0:3] = self.motionModel.sampleNewPose(ut, self.particles[i], deltaT)

    # Update the weight of each particle given a list of measurements
    def measurementUpdate(self, measurements):
        for i in range(self.numParticles):
            particle = self.particles[i]
            if self._outOfMap(particle):
                weight = 0
            else:
                weight = self.measurementModel.calculateProb(measurements, particle, self.grid, self.resolution)
            self.particles[i][3] *= weight

    # Resample all the particles based on their current weights.
    def resample(self):
        normalizer = 0
        for i in range(self.numParticles):
            normalizer += self.particles[i][3]

        for i in range(self.numParticles):
            self.particles[i][3] = self.particles[i][3] / normalizer

        newParticles = []
        step = 1.0 / self.numParticles
        r = random.uniform(0.0, step)
        c = self.particles[0][3]
        i = 0

        for m in range(self.numParticles):
            U = r + m * step
            while U > c:
                i = i + 1
                c = c + self.particles[i][3]
            newParticle = np.array(self.particles[i], copy=True)
            newParticle[3] = 1
            newParticles.append(newParticle)

        self.particles = newParticles

    # Calculate the estimated pose by averaging all the particles.
    def calculateEstimatedPose(self):
        totalX = 0
        totalY = 0
        totalCos = 0
        totalSin = 0

        for i in range(self.numParticles):
            totalX += self.particles[i][0]
            totalY += self.particles[i][1]
            totalCos += math.cos(self.particles[i][2])
            totalSin += math.sin(self.particles[i][2])

        x = totalX / self.numParticles
        y = totalY / self.numParticles
        theta = math.atan2(totalSin / self.numParticles, totalCos / self.numParticles)

        return np.array([x, y, theta])

    def getParticles(self):
        return self.particles

    def _outOfMap(self, particle):
        height, width = self.grid.shape
        x = particle[0]
        y = particle[1]
        if x < 0 or x > width * self.resolution or y < 0 or y > height * self.resolution:
            return True
        else:
            return False