import math
import numpy as np
import numpy.linalg as la
import feature as f


'''
Particle in the FastSLAM.
'''
class particle:
    # pose: Initial pose of the particle.
    # motionModel: Motion model for motion update.
    # measurementNoiseCov: Noise covariance of the measurement noise.
    # perceptionRange: Range this robot can perceive, used to reject outlier features.
    def __init__(self, pose, motionModel, measurementNoiseCov, perceptionRange, newFeatureWeight):
        self.pose = pose
        self.motionModel = motionModel
        self.measurementNoiseCov = measurementNoiseCov
        self.perceptionRange = perceptionRange

        # We store weight as negative logarithmic.
        self.newFeatureWeight = -math.log(newFeatureWeight)
        self.importanceWeight = 0

        # Measurement that has correspondance likelihood lower than
        # this threshold is considered a new feature.
        self.correspondanceLikelihoodThreshold = 0.01
        self.features = []
        # A path of past poses
        self.path = [pose]

    def motionUpdate(self, command, deltaT):
        self.pose = self.motionModel.sampleNewPose(command, self.pose, deltaT)
        self.path.append(self.pose)

    # Incorporate feature measurements. Determine if the measurements are previously
    # observed features or new features.
    def measurementUpdate(self, measurements):
        matchedFeaturesIdx = []

        for i in range(len(measurements)):
            z = measurements[i]
            featureIdx, likelihood, m_cov = self._findFeatureWithMaximumCorrespondence(z, matchedFeaturesIdx)

            # If this measurement is new.
            if likelihood < self.correspondanceLikelihoodThreshold:
                newFeature = self._initializeFeature(self.pose, z)
                self.features.append(newFeature)
                matchedFeaturesIdx.append(len(self.features) - 1)
                self.importanceWeight += self.newFeatureWeight
            # This measurement is observed before.
            else:
                # Observed Feature
                fPos = self.features[featureIdx].getPosition()
                mu = np.array([fPos]).transpose()
                cov = self.features[featureIdx].getCovariance()
                z_hat = self._measurementPrediction(self.pose, fPos)
                z_inov = np.array([z - z_hat]).transpose()

                H = self._calculateMeasurementJacobian(self.pose, fPos)
                # Kalman gain
                K = np.dot(cov, np.dot(H.transpose(), la.inv(m_cov)))

                # Update the estimate of the feature.
                mu_new = mu + np.dot(K, z_inov)
                cov_new = np.dot((np.identity(2) - np.dot(K, H)), cov)

                self.features[featureIdx].setPosition(mu_new.transpose()[0])
                self.features[featureIdx].setCovariance(cov_new)
                self.features[featureIdx].incrementCounter()

                matchedFeaturesIdx.append(featureIdx)
                self.importanceWeight += -math.log(likelihood)

        self._removeOutlierFeatures(matchedFeaturesIdx)



    # Remove the outlier features that is supposed to be observed with counter number of 0.
    def _removeOutlierFeatures(self, matchedFeaturesIdx):
        perceptionRangeSq = self.perceptionRange ** 2

        i = 0
        while i < len(self.features):
            if i not in matchedFeaturesIdx:
                fPosition = self.features[i].getPosition()
                distSq = (fPosition[0] - self.pose[0]) ** 2 + (fPosition[1] - self.pose[1]) ** 2

                # Features that are within perception range but not matched.
                if distSq < perceptionRangeSq:
                    self.features[i].decrementCounter()
                    if self.features[i].getCounter() <= 0:
                        del self.features[i]
                        continue
            i += 1


    # Calculate the jacobian matrix of measurement model.
    # robotPose: [x, y, theta]
    # featurePosition: [x, y]
    def _calculateMeasurementJacobian(self, robotPose, featurePosition):
        rp = robotPose
        fp = featurePosition
        sqDist = (robotPose[0] - featurePosition[0]) ** 2 + (robotPose[1] - featurePosition[1]) ** 2
        dist = math.sqrt(sqDist)

        return np.array([
            [(rp[1] - fp[1]) / sqDist, (fp[0] - rp[0]) / sqDist],
            [(fp[0] - rp[0]) / dist, (fp[1] - rp[1]) / dist]
        ])

    # Predict the measurement given robot pose and feature position.
    def _measurementPrediction(self, robotPose, featurePosition):
        fx = featurePosition[0]
        fy = featurePosition[1]
        rx = robotPose[0]
        ry = robotPose[1]
        rTheta = robotPose[2]
        dx = fx - rx
        dy = fy - ry
        return np.array([
            (math.atan2(dy, dx) - rTheta) % (2 * math.pi),
            math.sqrt(dx ** 2 + dy ** 2)
        ])

    # Initialize the position and covariance of the feature.
    def _initializeFeature(self, robotPose, measurement):
        rx = robotPose[0]
        ry = robotPose[1]
        rTheta = robotPose[2]
        mTheta = measurement[0]
        mDist = measurement[1]

        # Add measurement relative position over robot position.
        position = np.array([
            rx + mDist * math.cos(mTheta + rTheta),
            ry + mDist * math.sin(mTheta + rTheta)
        ])

        H = self._calculateMeasurementJacobian(robotPose, position)
        H_inv = la.inv(H)

        # Transform measurement space covariance to state space.
        cov = np.dot(H_inv, np.dot(self.measurementNoiseCov, H_inv.transpose()))

        return f.feature(position, cov)

    # Find the feature that corresponds to the measurement with maximum likelihood.
    # return [feature index, likelihood of the correspondence]
    def _findFeatureWithMaximumCorrespondence(self, measurement, matchedFeaturesIdx):
        maxLikelihood = 0
        featureIdx = -1
        measurementCov = np.identity(2)

        # Wrap measurement array to 2D
        z = np.array([measurement])

        for i in range(len(self.features)):
            if i in matchedFeaturesIdx:
                continue

            feature = self.features[i]
            # Expected measurement.
            z_hat = np.array([self._measurementPrediction(self.pose, feature.getPosition())])
            # Innovation vector.
            z_inov = (z - z_hat).transpose()

            H = self._calculateMeasurementJacobian(self.pose, feature.getPosition())

            # Measurement covariance
            m_cov = np.dot(H, np.dot(feature.getCovariance(), H.transpose())) + self.measurementNoiseCov

            # Likelihood of correspondence.
            w = 1.0 / math.sqrt(np.fabs(la.det(2 * math.pi * m_cov))) \
                * np.exp(- 1.0 / 2.0 * np.dot(z_inov.transpose(), np.dot(la.inv(m_cov), z_inov))[0, 0])

            # w is not suppose to be larger than 1, but somehow it is.
            # w = min(1.0, w)

            if w > maxLikelihood:
                maxLikelihood = w
                featureIdx = i
                measurementCov = m_cov

        return featureIdx, maxLikelihood, measurementCov

    def getPose(self):
        return self.pose

    def setPose(self, pose):
        self.pose = pose

    def getPath(self):
        return self.path

    def addFeature(self, feature):
        self.features.append(feature)

    def getFeatures(self):
        return self.features

    def getWeight(self):
        return math.exp(-self.importanceWeight)

    def setWeight(self, weight):
        self.importanceWeight = -math.log(weight)
