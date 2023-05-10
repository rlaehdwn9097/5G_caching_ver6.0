import sys
sys.path.insert(0,'C:/Users/USER/vscodeworkspace/5G_Caching_porting_new_env')

import NetworkEnv.config as cf
from NetworkEnv import network as nt
from ReplayBuffer import ReplayBuffer
from NetworkEnv.scenario import Scenario

#import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import gym
import argparse
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float64')
#wandb.init(name='DQN_a1.5_b2', project="cache_sim")

parser = argparse.ArgumentParser()
parser.add_argument('--contentfile', type=str, default=cf.CONTENTFILE)
parser.add_argument('--generating_method', type=str, default=cf.GEN_METHOD)

args = parser.parse_args()

class ActionStateModel:
    def __init__(self, state_dim, action_space):
        self.state_dim  = state_dim
        self.action_dim = len(action_space)
        self.action_space = action_space
        self.epsilon = cf.EPSILON
        
        self.model = self.create_model()
        self.model.summary()

        self.random_action_cnt_list = [0,0,0,0]
        self.qs_action_cnt_list = [0,0,0,0]
    
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
        state = np.reshape(state, [1, self.state_dim])
        #self.epsilon *= args.eps_decay
        #self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        
        if np.random.random() < self.epsilon:
            action = random.randint(0,self.action_dim - 1)
            #print('random action : {}'.format(action))
            self.random_action_cnt_list[action] += 1
            return action
        
        action = np.argmax(q_value)
        self.qs_action_cnt_list[action] += 1
        return action

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)
    

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
    
    def train_model(self):
        #if self.buffer.buffer_count() >= cf.BUFFER_SIZE:
        #    return
        
        # buffer 에서 sample 해서 s, a, r, s', d 가져옴
        states, actions, rewards, next_states, done = self.buffer.sample_batch(cf.BATCH_SIZE)
        # target_value
        targets = self.target_model.predict(states)
        next_q_values = self.target_model.predict(next_states).max(axis=1)

        # td_target
        targets[range(cf.BATCH_SIZE), actions] = rewards + (1-done) * next_q_values * cf.GAMMA
        '''
        for i in range(32):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                t_targets[i][actions[i]] = rewards[i]
            else:
                t_targets[i][actions[i]] = rewards[i] + args.gamma * (np.amax(next_q_values[i]))
        '''

        self.model.train(states, targets)
    
    def train(self, max_episodes=2000):
        for ep in range(max_episodes):
            #@ 1. 변수와 network reset 해줌.
            done, episode_reward = False, 0
            #
            self.model.random_action_cnt_list = [0,0,0,0]
            self.model.qs_action_cnt_list = [0,0,0,0]
            self.env.reset_parameters()
            while not done:
                CACHE_HIT_FLAG, state, requested_content, path, done = self.env.run_round()
                
                #@ 6. core network 에서 cache hit이 일어났을 때
                if CACHE_HIT_FLAG == 0:
                    
                    #@ 7. action 선택
                    action = self.model.get_action(state)
                    next_state, reward = self.env.step(action, path, requested_content)
                    episode_reward += reward
                    self.buffer.store(state, action, reward, next_state, done)
                    #@ 8. buffer에 어느정도 찼을 때, DQN train 시작. target network update
                    if self.buffer.buffer_count() >= cf.BUFFER_SIZE/2:
                        if self.model.epsilon > cf.EPSILON_MIN:
                            self.model.epsilon *= cf.EPSILON_DECAY
                            
                        self.train_model()
                        self.target_softupdate()
                        
                #@ 9. cache hit 이 일어났을 경우 content storage update만 진행.
                else:
                    self.env.update()

            #self.target_update()
            #@ ep 끝나는 라인
            self.env.printResults(ep,episode_reward, self.model.random_action_cnt_list, self.model.qs_action_cnt_list)
            #wandb.log({'Reward': episode_reward})
            #wandb.log({'CHR': self.env.total_cache_hit_count/self.env.round_nb})
            #wandb.log({'Avg_hop': self.env.total_hop/self.env.round_nb})

def main():
    scenario = Scenario(args)
    env = nt.Network(scenario,args)
    agent = Agent(env)
    agent.train()

if __name__ == "__main__":
    main()
    