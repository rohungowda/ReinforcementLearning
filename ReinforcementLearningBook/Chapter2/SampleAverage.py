import numpy as np
import random
import matplotlib.pyplot as plt
import math
from Distributions import Distributions, StationaryDistribution, NonStationaryDistribution
from Policy import Policy, EpsilonGreedyPolicy, UpperConfidencePolicy, SoftmaxPolicy
from Updates import Updates, StationaryUpdates, DynamicUpdates, optimalStepSizeUpdate, gradientUpdate

class Agent:
    def setPolicy(self, policy: Policy):
        self.policy = policy

    def setUpdate(self, update: Updates):
        self.update = update



def BanditProblem(distribution: Distributions, agent: Agent, timesteps: int):

    rewardsPerTimestep = []

    for t in range(timesteps):

        distribution.update()

        state = agent.update.ReturnState(t)
        actionIndex = agent.policy.calculatePolicy(state)
 
        # Calculate Current Rewards for choosing action
        currentReward = distribution.calculateReward(actionIndex)

        # append rewards
        rewardsPerTimestep.append(currentReward)

        agent.update.update(currentReward,actionIndex)


    return np.array(rewardsPerTimestep)


Runs = 2000
Actions = 10
VarianceLimit = 1.0
epsilon = 0.1
timestep = 3_000
stepSizeParameter = 0.1
c = 2.0


plt.figure(figsize=(10,6))  # Create one figure for all lines

agent = Agent()

BanditProblemInfo = {
    "gradient": [SoftmaxPolicy(c), gradientUpdate(Actions, stepSizeParameter)],
    "upperConfidence":[UpperConfidencePolicy(c),optimalStepSizeUpdate(Actions, stepSizeParameter)],
    "optimalStepSize":[EpsilonGreedyPolicy(epsilon), optimalStepSizeUpdate(Actions, stepSizeParameter)],
    "dynamic":[EpsilonGreedyPolicy(epsilon),DynamicUpdates(Actions, stepSizeParameter)],
    "static":[EpsilonGreedyPolicy(epsilon),StationaryUpdates(Actions)]
}

MetricsInfo = {
    "gradient": np.zeros(timestep),
    "upperConfidence":np.zeros(timestep),
    "optimalStepSize":np.zeros(timestep),
    "dynamic":np.zeros(timestep),
    "static":np.zeros(timestep)
}


for run in range(Runs):
    if run % 100 == 0:
        print("*"*20 + f"Log {run}" + "*" * 20)
    
    for name,(policy,update) in BanditProblemInfo.items():
        distribution = NonStationaryDistribution(Actions, VarianceLimit, startingMean=4.0, startingVariance=1.0)
        #distribution = StationaryDistribution(Actions,VarianceLimit)
        if run % 100 == 0:
            print("*"*10 + name + "*"*10)
        agent.setPolicy(policy)
        agent.setUpdate(update)
        rewards = BanditProblem(distribution, agent, timestep)
        MetricsInfo[name] += rewards

        update.Reset()

for index,name in enumerate(MetricsInfo.keys()):
    MetricsInfo[name] /= Runs

    plt.plot(
        np.arange(timestep),
        MetricsInfo[name],
        label=f"{name}"
    )

print("Succefully Finished All Runs, creating plot...")
plt.xlabel('Timesteps')
plt.ylabel('Average Reward')
plt.title(f"Average Rewards over Different Algorithms")

plt.legend()
plt.grid(True)

plt.savefig(f"./Plots/average_rewards_nonStationary.png")
