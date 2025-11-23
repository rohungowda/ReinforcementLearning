import numpy as np
from dataclasses import dataclass

@dataclass
class State:
    timestamp: int
    valuePredictions: np.array
    totalActionTaken: np.array

def softmax(x):
    exponents = np.exp(x - np.max(x)) # to stop from causing overflow essentially scaling down, relationship is still the same
    return exponents / np.sum(exponents)