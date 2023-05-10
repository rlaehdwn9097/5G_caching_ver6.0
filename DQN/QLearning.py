import sys
import csv
sys.path.insert(0,'C:/Users/FNC/VscodeWorkSpace/For_QLearning/paper')

import NetworkEnv.config as cf
import NetworkEnv.Network as nt
from ReplayBuffer import ReplayBuffer

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float64')

class Q_Learning_Agent:
    def __init__(self, env):
        self.env = env
        #self.state_dim = self.env.observation_space.shape[0]
        self.state_dim = env.state_dim
        self.action_space = self.env.action_space

        # Hyperparameters
        self.alpha = 0.0001  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.9997  # Exploration decay
        self.num_episodes = 10000  # Number of episodes to run

        self.random_action_cnt_list = [0,0,0,0]
        self.qs_action_cnt_list = [0,0,0,0]
    

        # Initialize the Q-table
        #print(self.state_dim, self.action_space)
        self.q_table = np.zeros((20000, len(self.action_space)), dtype='float32')
        #print(self.q_table.shape)
    
    def save_q_table(self, q_table, file_name):
        np.save(file_name, q_table)

    def load_q_table(self, file_name):
        return np.load(file_name)
    
    def train(self, max_episodes=20000):

        # Q-learning algorithm
        reward_list = []
        for episode in range(max_episodes):
            state = self.env.reset()
            self.env.reset_parameters()
            self.random_action_cnt_list = [0,0,0,0]
            self.qs_action_cnt_list = [0,0,0,0]
            episode_reward, reward, done = 0, 0, False

            while not done:
                CACHE_HIT_FLAG, state, requested_content, path, done = self.env.run_round()

                if CACHE_HIT_FLAG == 0:
                    if np.random.uniform(0, 1) < self.epsilon:
                        action = random.randint(0,3)  # Explore
                        self.random_action_cnt_list[action] += 1
                        #print('radom action : ', action)
                    else:
                        #print(self.q_table)
                        #print(self.q_table[state,:])
                        #print(np.argmax(self.q_table[state,:]))
                        #! QS ACTION 뽑은 거 문제.
                        index = int(np.argmax(self.q_table[state])/4)
                        action = np.argmax(self.q_table[state][index])
                        #action = np.argmax(np.argmax(self.q_table[state]))  # Exploit
                        self.qs_action_cnt_list[action] += 1
                        #print('qs_action : ',action)
                        #print('exploit action : ', action)

                    
                    
                    next_state, reward = self.env.step(action, path, requested_content)
                    episode_reward+= reward
                    
                    #print(len(state))
                    #print(len(next_state))
                    #print(self.q_table[next_state])
                    #print(self.q_table[state, action])
                    # Update the Q-table
                    self.q_table[state, action] += self.alpha * (
                        reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action]
                    )
                    state = next_state
                else:
                    self.env.update()

            # Decay the exploration rate
            self.epsilon *= self.epsilon_decay
            print('qs_action_list : ',self.qs_action_cnt_list)
            print('random_action_list : ',self.random_action_cnt_list)
            self.env.printResults(episode, episode_reward, self.random_action_cnt_list, self.qs_action_cnt_list)
            


def main():
    env = nt.Network()
    agent = Q_Learning_Agent(env)
    agent.train()

if __name__ == "__main__":
    main()


'''
# Test the learned Q-table
num_test_episodes = 10
num_steps = 100
for episode in range(num_test_episodes):
    state = env.reset()
    done = False

    print(f"Episode {episode + 1}:")
    env.render()
    for step in range(num_steps):
        action = np.argmax(q_table[state])
        state, _, done, _ = env.step(action)
        env.render()
        if done:
            break
save_q_table(q_table, 'q_table.npy')
'''

