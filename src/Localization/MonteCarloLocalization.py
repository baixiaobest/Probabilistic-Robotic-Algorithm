import random
import math
import numpy as np


class MonteCarloLocalization:
    # grid: Occupancy grid.
    # poseGuess: Initial guess of robot position. [XMin, XMax, YMin, YMax].
    # numParticles: Number of particles used.
    # motionModel: Probabilistic motion model.
    # measurementModel: Probabilistic measurement model.
    # enableParticleInjection: Inject random particle in case of localization failure.
    # alphaSlow: Parameter that make avgWeightSlow slowly follows average weight of the particles.
    # alphaFast: Parameter that make avgWeightFast quickly follows average weight of the particles.
    def __init__(self,
                 grid,
                 resolution,
                 poseGuess,
                 numParticles,
                 motionModel,
                 measurementModel,
                 enableParticleInjection=False,
                 alphaSlow=0.1,
                 alphaFast=1):
        self.grid = grid
        self.resolution = resolution
        self.numParticles = numParticles
        self.motionModel = motionModel
        self.measurementModel = measurementModel
        self.enableParticleInjection = enableParticleInjection
        self.alphaSlow = alphaSlow
        self.alphaFast = alphaFast
        self.particles = []
        self.avgWeightSlow = 0
        self.avgWeightFast = 0

        self.xMin = poseGuess[0]
        self.xMax = poseGuess[1]
        self.yMin = poseGuess[2]
        self.yMax = poseGuess[3]

        self.particles = self._generateRandomParticles(numParticles)

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

        # Determine number of injected random new particles.
        numParticlesInject = 0
        if self.enableParticleInjection:
            avgWeight = normalizer / self.numParticles
            self.avgWeightSlow += self.alphaSlow * (avgWeight - self.avgWeightSlow)
            self.avgWeightFast += self.alphaFast * (avgWeight - self.avgWeightFast)
            numParticlesInject = int(np.fmax(0, 1.0 - self.avgWeightFast / self.avgWeightSlow) * self.numParticles)

        # Inject new random particles
        newParticles = self._generateRandomParticles(numParticlesInject)

        # Start resampling from old particles
        numParticlesToResample = self.numParticles - numParticlesInject
        step = 1.0 / (numParticlesToResample)
        r = random.uniform(0.0, step)
        c = self.particles[0][3]
        i = 0

        for m in range(numParticlesToResample):
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

    # Generate a set of particles.
    # Particle is represented as [x, y, theta, weight]
    def _generateRandomParticles(self, numParticles):
        particles = []

        for i in range(numParticles):
            x = random.uniform(self.xMin, self.xMax)
            y = random.uniform(self.yMin, self.yMax)
            theta = random.uniform(0, 2.0 * math.pi)
            particles.append(np.array([x, y, theta, 1]))
        return particles

    def _outOfMap(self, particle):
        height, width = self.grid.shape
        x = particle[0]
        y = particle[1]
        if x < 0 or x > width * self.resolution or y < 0 or y > height * self.resolution:
            return True
        else:
            return False