import random as r

# Sample a normal distribution with standard deviation of sigma
def sampleNormal(mu, sigma):
    total = 0
    for i in range(12):
        total += r.uniform(-sigma, sigma)
    return total / 2 + mu
