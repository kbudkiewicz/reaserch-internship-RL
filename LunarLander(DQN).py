import random

import torch
import torch.nn as nn
import torch.nn.functional as tfunc
import math
import matplotlib.pyplot as mplt
import numpy
import gym
from collections import deque, namedtuple
import Box2D

### Assigning the device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Current device: %s \n" % device)

### Creating gym' Lunar Lander environment
env = gym.make("LunarLander-v2", new_step_api=True)
total_reward = 0.0
total_steps = 0
obs = env.reset()
print(env.step(1))

### Loop testing whether a condition in the environment (both legs touching the ground // True) is met. Then break the loop and return total score
while True:
    action = env.action_space.sample()
    obs, reward, f1 ,f2, _ = env.step(action)
    total_reward += reward
    total_steps += 1
    if f1 and f2 == True:
        break
print("Episode done in %d steps. Total score: %s" % (total_steps,total_reward))

# nn.BatchNorm2d()

class DQN(nn.Module):
    def __init__(self):                         # CNN not needed for research internship -> Linear layers
        super(DQN,self).__init__()              # super(superclass) - inherit the methods of the superclass (class above this one). Here: inherit all __init__ method of DQN
        self.lin1 = nn.Linear(8,32)             # input (here 8) corresponds to the size of observation space
        self.lin2 = nn.Linear(32,32)
        self.lin3 = nn.Linear(32,4)             # output (here 4) corresponds to the size of action space

        ### For CNNs
        # stide         gives how much the filter is moved across the matrix (e.g. stride = 2 means: move the filter 2 indicies to the right of the matrix)
        # kernel_size   size of the tensor/matrix filter between convolutional layers

    def forward(self,state):
        x = tfunc.relu(self.lin1(state))               # ReLU - rectified linear unit. take max(0,input) of the input
        x = tfunc.relu(self.lin2(x))
        return tfunc.relu(self.lin3(x))

### Defining replay memory
transit = namedtuple('Transition',('s_0','a_0','r_0','s_1'))

class Replay_memory(object):
    def __init__(self,mem_size):
        self.memory = deque(mem_size)
    def push(self,args):
        self.memory.append(transit(args))
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)

class Environment():
    def __init__(self):
        self.steps_left = 100

    def get_obs(self):
        return env.reset()

    def get_action(self):
        return env.action_space.sample()

    def is_done(self):
        return self.steps_left == 0

    def action(self, action):
        if self.is_done():
            return "Testing is over"
        self.steps_left -= 1
        return numpy.random.random()

class Agent():
    def __init__(self):
        self.score = 0.0

    def step(self, env):
        current_obs = env.observation_space().sample()
        actions = env.action_space().sample()
        reward = env.action(numpy.random.choice(actions))
        self.score += reward

### Training
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# def select_action(state):
    # sample = random.random()
    # eps_treshold = EPS_END + (EPS_START-EPS_END)* \ math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    # if sample > eps_treshold:
    #     return policy_net(state).max(1)[1].view(1,1)
    # else:
    #     return torch.tensor([[random.randrange(n_actions)]], device=device,dtype=torch.long)

### Training loop
# episode_n = 10000
# scores = []
# for i_episode in range(n_episodes):
#     state = env.reset()
#     score = 0
#     for z in range(max_t):
#         action = agent.act(state,eps)
#         next_state, reward, done, _ = env.step(action)
#         agent.step(step, action, reward, next_state, done)
    #     state = next_state
#         score += reward
    #     if done:
#             break
#     scores.append(score)
#     if np.mean(scores) >= 200.00:
#       print("Environment solved in %s iterations with avarage score %s" % (i_episode, np.mean(scores))
#   return scores

# env.render()
# env.close()