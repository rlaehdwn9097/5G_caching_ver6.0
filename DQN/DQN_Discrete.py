import sys
sys.path.insert(0,'C:/Users/USER/VScodeWorkspace/DeepRL-TensorFlow2')

import NetworkEnv.config as cf
import NetworkEnv.Network as nt
from ReplayBuffer import ReplayBuffer

import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import gym
import argparse
import numpy as np
from collections import deque
import random

#tf.keras.backend.set_floatx('float64')
wandb.init(name='DQN13_1111', project="dqn_diff")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()
'''
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
'''

class ActionStateModel():
    def __init__(self, state_dim, action_space):
        self.state_dim  = state_dim
        self.action_dim = len(action_space)
        self.action_space = action_space
        self.epsilon = cf.EPSILON
        
        self.model = self.create_model()
        self.model.summary()
    
    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim*cf.H1,)),
            Dense(self.state_dim*cf.H2, activation='relu'),
            Dense(self.state_dim*cf.H8, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])
        model.compile(loss='mse', optimizer=Adam(cf.LEARNING_RATE))
        return model
    
    def predict(self, state):
        return self.model.predict(state)
    
    def get_action(self, state):
        if np.random.random() <= self.epsilon:
            action = int(np.random.choice(self.action_space,1))
            #self.random_action_cnt_list[action] += 1
            return action
        else:
            qs = self.model.predict(tf.convert_to_tensor([state],dtype=tf.float32))
            action = np.argmax(qs)
            #self.qs_action_cnt_list[action] += 1
            return action

    def train(self, state, actions, td_targets):
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(actions, self.action_dim)
            q = self.model(state, training=True)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            loss = tf.reduce_mean(tf.square(q_values-td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save_weights(self):
        self.model.save_weights()
    

class Agent:
    def __init__(self, env):
        self.env = env
        #self.state_dim = self.env.observation_space.shape[0]
        self.state_dim = env.state_dim
        self.action_space = self.env.action_space

        self.model = ActionStateModel(self.state_dim, self.action_space)
        self.target_model = ActionStateModel(self.state_dim, self.action_space)
        self.target_update()

        self.buffer = ReplayBuffer(cf.BUFFER_SIZE)

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def target_softupdate(self):
        phi = self.model.model.get_weights()
        target_phi = self.target_model.model.get_weights()

        for i in range(len(phi)):
            target_phi[i] = cf.TAU * phi[i] + (1 - cf.TAU) * target_phi[i]
        self.target_model.model.set_weights(target_phi)

    def td_target(self, rewards, target_qs, dones):
        max_q = np.max(target_qs, axis=1, keepdims=True)
        y_k = np.zeros(max_q.shape)
        for i in range(max_q.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + cf.GAMMA * max_q[i]
        return y_k
    
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + (1-done) * next_q_values * args.gamma
            self.model.train(states, targets)
    
    def train(self, max_episodes=cf.MAX_EPISODE_NUM):
        
        for ep in range(max_episodes):
            #@ 1. 변수와 network reset 해줌.
            done, episode_reward = False, 0
            self.env.reset_parameters()
            while not done:
                CACHE_HIT_FLAG, state, requested_content, path, done = self.env.run_round()
                #@ 6. data center 와 core network 에서 cache hit이 일어났을 때
                if CACHE_HIT_FLAG == 0:
                    #@ 7. action 선택
                    action = self.model.get_action(state)
                    next_state, reward = self.env.step(action, path, requested_content)
                    episode_reward += reward
                    self.buffer.store(state, action, reward, next_state, done)
                    #@ 8. buffer에 어느정도 찼을 때, DQN train 시작. target network update
                    if self.buffer.buffer_count() > cf.BUFFER_SIZE/2:
                        #print('buffer count half full')
                        if self.model.epsilon > cf.EPSILON_MIN:
                            self.model.epsilon *= cf.EPSILON_DECAY
                        # sample transitions from replay buffer
                        states, actions, rewards, next_states, dones = self.buffer.sample_batch(cf.BATCH_SIZE)
                        # predict target Q-values
                        target_qs = self.target_model.predict(tf.convert_to_tensor(next_states, dtype=tf.float32))
                        # compute TD targets
                        y_i = self.td_target(rewards, target_qs, dones)
                        self.model.train(tf.convert_to_tensor(states, dtype=tf.float32), actions, tf.convert_to_tensor(y_i, dtype=tf.float32))
                        #print('after train')
                        # update target network(soft update)
                        self.target_softupdate()
                #@ 9. cache hit 이 일어났을 경우 content storage update만 진행.
                else:
                    self.env.update()
                
            #self.target_update()
            #if ep % 50 == 0:
                self.target_update()
            #@ ep 끝나는 라인
            self.env.printResults(ep,episode_reward)
            if ep % 50 == 0:
                self.model.model.save_weights("./DQN/DQN_weights/dqn_{}.h5".format(ep))




def main():
    # Enviorment
    env = nt.Network()
    agent = Agent(env)
    agent.train()

if __name__ == "__main__":
    main()
    