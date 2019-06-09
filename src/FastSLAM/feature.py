
'''
A feature in side a FastSLAM particle.
'''
class feature:
    # position: Estimated position of the particle, [x, y]
    # covariance: covariance matrix of the estimated position.
    def __init__(self, position, covariance):
        self.position = position
        self.covariance = covariance
        # Counter that keep track of how many times feature is observed.
        # When it is observed, increment this counter,
        # when it is not observed but it should have, decrement this counter.
        self.counter = 1

    def getPosition(self):
        return self.position

    def setPosition(self, position):
        self.position = position

    def getCovariance(self):
        return self.covariance

    def setCovariance(self, covariance):
        self.covariance = covariance

    def incrementCounter(self):
        self.counter += 1

    def decrementCounter(self):
        self.counter -= 1

    def getCounter(self):
        return self.counter