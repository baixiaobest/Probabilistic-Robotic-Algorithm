
'''
A feature in side a FastSLAM particle.
'''
class feature:
    # position: Estimated position of the particle, [x, y]
    # covariance: covariance matrix of the estimated position.
    def __init__(self, position, covariance):
        self.position = position
        self.covariance = covariance

    def getPosition(self):
        return self.position

    def setPosition(self, position):
        self.position = position

    def getCovariance(self):
        return self.covariance

    def setCovariance(self, covariance):
        self.covariance = covariance