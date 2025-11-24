from abc import ABC, abstractmethod
from helpers import softmax
import numpy as np
import math
from helpers import State, softmax


class Policy(ABC):
    
    @abstractmethod
    def calculatePolicy(self,state: State):
        pass

class SoftmaxPolicy(Policy):

    def calculatePolicy(self, state: State):
        t, valuePredictions, totalActionTaken = state.timestamp, state.valuePredictions, state.totalActionTaken
        return np.random.choice(len(valuePredictions), p=softmax(valuePredictions))

class UpperConfidencePolicy(Policy):
    def __init__(self, c):
        self.c = c

    def calculatePolicy(self, state):
        t, valuePredictions, totalActionTaken = state.timestamp, state.valuePredictions, state.totalActionTaken
        return np.argmax([
            (valuePrediction + (self.c * math.sqrt(math.log(t)/totalActionTaken[index])))  if totalActionTaken[index] != 0
            else float('inf')
            for index,valuePrediction in enumerate(valuePredictions)])


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def calculatePolicy(self, state):
        t, valuePredictions, totalActionTaken = state.timestamp, state.valuePredictions, state.totalActionTaken
        greedyAction = np.argmax(valuePredictions)
        nonGreedyAction = np.random.choice(len(valuePredictions))
        return np.random.choice([greedyAction, nonGreedyAction], p=[1-self.epsilon, self.epsilon])
