from abc import ABC, abstractmethod
import numpy as np
from helpers import State, softmax

class Updates:
    def __init__(self, numberActions):
        self.numberActions = numberActions
        self.valuePredictions = np.zeros(numberActions)
        self.numberTimesActionSeen = np.zeros(numberActions)

    def update(self,currentReward, actionIndex):
        raise NotImplementedError()
    
    def ReturnState(self, timestamp: int):
        return State(timestamp,self.valuePredictions, self.numberTimesActionSeen)
    
    def Reset(self):
        raise NotImplementedError()


class gradientUpdate(Updates):
    def Reset(self):
        self.valuePredictions = np.zeros(self.numberActions)
        self.baselineRewards = np.zeros(self.numberActions)
        self.numberTimesActionSeen = np.zeros(self.numberActions)

    def __init__(self, numberActions, stepSizeParameter):
        super().__init__(numberActions)
        self.stepSizeParameter = stepSizeParameter
        self.Reset()
        

    def update(self, currentReward, actionIndex):
        self.numberTimesActionSeen[actionIndex] += 1
        self.valuePredictions = self.valuePredictions + self.stepSizeParameter * ((currentReward - self.baselineRewards)*((np.eye(1,self.numberActions,actionIndex).flatten()) - softmax(self.valuePredictions)))
        self.baselineRewards[actionIndex] = ((1 - self.stepSizeParameter) *  self.baselineRewards[actionIndex]) + (currentReward * self.stepSizeParameter)


class optimalStepSizeUpdate(Updates):

    def Reset(self):
        self.valuePredictions = np.zeros(self.numberActions)
        self.numberTimesActionSeen = np.zeros(self.numberActions)
        self.parameter_denominator = 0

    def __init__(self, numberActions, stepSizeParameter):
        super().__init__(numberActions)
        self.stepSizeParameter = stepSizeParameter
        self.Reset()

    def update(self, currentReward, actionIndex):
        self.numberTimesActionSeen[actionIndex] += 1
        self.parameter_denominator = self.parameter_denominator + self.stepSizeParameter * (1 - self.parameter_denominator)
        Beta = self.stepSizeParameter / self.parameter_denominator
        self.valuePredictions[actionIndex] = ((1 - Beta) * self.valuePredictions[actionIndex]) + (Beta * currentReward)

class DynamicUpdates(Updates):
    
    def Reset(self):
        self.valuePredictions = np.zeros(self.numberActions)
        self.numberTimesActionSeen = np.zeros(self.numberActions)

    def __init__(self,numberActions, stepSizeParameter):
        super().__init__(numberActions)
        self.stepSizeParameter = stepSizeParameter
        self.Reset()


    def update(self,currentReward, actionIndex):
        self.numberTimesActionSeen[actionIndex] += 1
        self.valuePredictions[actionIndex] = ((1 - self.stepSizeParameter) *  self.valuePredictions[actionIndex]) + (currentReward * self.stepSizeParameter)


class StationaryUpdates(Updates):
    
    def Reset(self):
        self.valuePredictions = np.zeros(self.numberActions)
        self.numberTimesActionSeen = np.zeros(self.numberActions)

    def __init__(self,numberActions):
        super().__init__(numberActions)
        self.Reset()


    def update(self,currentReward, actionIndex):
        self.numberTimesActionSeen[actionIndex] += 1
        self.valuePredictions[actionIndex] = self.valuePredictions[actionIndex] + (1/ self.numberTimesActionSeen[actionIndex]) * (currentReward - self.valuePredictions[actionIndex])