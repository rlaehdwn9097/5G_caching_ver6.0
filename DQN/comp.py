#import NetworkEnv.config as cf
import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import gym
import argparse
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float64')
wandb.init(name='DQN15_666', project="dqn_diff")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps
        
        self.model = self.create_model()
    
    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse', optimizer=Adam(args.lr))
        return model
    
    def predict(self, state):
        return self.model.predict(state)
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)
    

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    
    def train_model(self):
        
        # buffer 에서 sample 해서 s, a, r, s', d 가져옴
        states, actions, rewards, next_states, done = self.buffer.sample()
        # target_value
        targets = self.target_model.predict(states)
        t_targets = targets
        next_q_values = self.target_model.predict(next_states).max(axis=1)

        # td_target
        targets[range(args.batch_size), actions] = rewards + (1-done) * next_q_values * args.gamma
        print(type(targets))
        print(targets.shape)
        print(targets[0])
        print(targets)
        

        print('=====VVV=====')

        for i in range(32):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                t_targets[i][actions[i]] = rewards[i]
            else:
                t_targets[i][actions[i]] = rewards[i] + args.gamma * (np.amax(next_q_values[i]))
        
        print(type(t_targets))
        print(t_targets.shape)
        print(t_targets[0])
        print(t_targets)

        self.model.train(states, targets)
    
    def train(self, max_episodes=500):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, reward*0.01, next_state, done)
                total_reward += reward
                state = next_state
            if self.buffer.size() >= args.batch_size:
                self.train_model()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            wandb.log({'Reward': total_reward})
            
            if ep%10 == 0:    
                self.target_update()


def main():
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.train()

if __name__ == "__main__":
    main()
    