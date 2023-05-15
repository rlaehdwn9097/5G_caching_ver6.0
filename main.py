# 원하는 Agent import
#from DQN.DQN15_Discrete import Agent
from DQN.QLearning import Agent
from NetworkEnv import network as nt
from NetworkEnv.scenario import Scenario
from NetworkEnv import config as cf

if __name__ == "__main__":
    scenario = Scenario()
    env = nt.Network(scenario)
    agent = Agent(env)
    agent.train()
    