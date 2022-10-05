import random
import torch
import torch.nn as nn
import torch.nn.functional as tfunc
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as mplt
import numpy
import math
import gym
from collections import deque, namedtuple
import Box2D

### Assigning the device
if torch.cuda.is_available():
    torch.device = "cuda"
else:
    torch.device = "cpu"
print("Current device: %s \n" % torch.device)

### Creating gym' Lunar Lander environment
env = gym.make("LunarLander-v2")
# obs = env.reset()
# print(obs,'\n', env.step(1))

### Loop testing whether a condition in the environment (both legs touching the ground // True) is met. Then break the loop and return total score
# total_reward = 0.0
# total_steps = 0
# while True:
#     action = env.action_space.sample()
#     obs, reward, f1 ,f2, _ = env.step(action)
#     total_reward += reward
#     total_steps += 1
#     if f1 and f2 == True:
#         break
# print("Episode done in %d steps. Total score: %s" % (total_steps,total_reward))

class DQN(nn.Module):
    def __init__(self,layer_size=32):                   # CNN not needed for research internship -> Linear layers, batchnormalisation not needed
        super(DQN,self).__init__()                      # super(superclass) - inherit the methods of the superclass (class above this one). Here: inherit all __init__ method of DQN
        self.lin1 = nn.Linear(8,layer_size)             # input (here 8) corresponds to the size of observation space
        self.lin2 = nn.Linear(layer_size,layer_size)
        self.lin3 = nn.Linear(layer_size,4)             # output (here 4) corresponds to the size of action space

        ### For CNNs
        # stide         gives how much the filter is moved across the matrix (e.g. stride = 2 means: move the filter 2 indicies to the right of the matrix)
        # kernel_size   size of the tensor/matrix filter between convolutional layers

    def forward(self,state):
        # x = state.to(torch.device)
        x = tfunc.relu(self.lin1(state))               # ReLU - rectified linear unit. take max(0,input) of the input
        x = tfunc.relu(self.lin2(x))
        return tfunc.relu(self.lin3(x))

### Defining replay memory
transit = namedtuple('Transition',('s_0','a_0','r_0','s_1'))

class Replay_memory(object):
    def __init__(self,memory_size):
        self.memory = deque(maxlen=memory_size)
    def push(self,*args):
        self.memory.append(transit(*args))
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self,):
        self.state_size = 8
        self.action_size = 4
        self.qnet_local = DQN().to(torch.device)
        self.qnet_target = DQN().to(torch.device)
        self.optimize = optim.SGD(self.qnet_local.parameters(), lr=5*10**-4)

        self.memory = Replay_memory(100)
        self.t_step = 0

    def step(self, *args):
        self.memory.push(*args)
        self.t_step = (self.t_step + 1) % TARGET_UPDATE
        if (self.t_step % TARGET_UPDATE == 0) and (self.memory == 100):
            self.learn( self.memory.sample(batch_size=10) )

    def learn(self, exp, GAMMA):
        s, a, r, next_s = exp
        q_target_next = self.qnet_target(next_s).detach().max(1)[0].unsqueeze(1)

        # Bellman equation
        q_target = r + GAMMA * q_target_next * (1-self.t_step)
        q_expect = self.qnet_local(s).gather(1, a)

        loss = tfunc.mse_loss(q_expect, q_target)
        self.optimize.zero_grad()
        loss.backward()
        self.optimize.step()

        self.update(self.qnet_local,self.qnet_target, TAU)

    def update(self, local, target, tau):
        for target, local in zip(target.parameters(), local.parameters()):
            target.data.copy_(tau*local.data + (1-tau)*target.data)

### Training parameters
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1*10**-3
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

##### Playground
# env.reset()
# n_network = DQN()
# n_network.forward()

### Input extraction
# state = env.reset()
# action = env.action_space.sample()
# n_actions = env.action_space.

### Training