import numpy as np

class Distributions:
    def __init__(self, numberActions, varianceLimit):
        self.numberActions = numberActions
        self.std = np.sqrt(np.random.uniform(0, varianceLimit))
    def update(self):
        pass
    def calculateReward(self, actionIndex):
        pass


class NonStationaryDistribution(Distributions):
    def __init__(self, numberActions, varianceLimit, startingMean=0.0, startingVariance=1.0):
        super().__init__(numberActions,varianceLimit)

        optimalStartingPoint = np.random.normal(startingMean, np.sqrt(startingVariance))
        self.optimalReward = np.array([ optimalStartingPoint for q_a in range(self.numberActions)])

    def update(self):
        self.optimalReward += np.array([np.random.normal(0,0.01) for q_a in range(self.numberActions)])

    def calculateReward(self, actionIndex):
        return np.random.normal(self.optimalReward[actionIndex], self.std)


class StationaryDistribution(Distributions):
    def __init__(self, numberActions, varianceLimit):
        super().__init__(numberActions,varianceLimit)

        self.actualExpectedRewards = [np.random.normal() for q_a in range(self.numberActions)]
        self.distributions = [(mean,self.std) for mean in self.actualExpectedRewards]
        
    def calculateReward(self, actionIndex):
        mean,deviation = self.distributions[actionIndex]
        return np.random.normal(mean,deviation)