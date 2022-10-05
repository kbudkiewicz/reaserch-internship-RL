import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tfunc
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as mplt
import gym
from collections import deque, namedtuple
import Box2D

### Assigning the device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Current device: %s \n" % device.upper())

### Creating gym' Lunar Lander environment
env = gym.make("LunarLander-v2")
env.reset(seed=0)
# obs = env.reset()
# print(obs,'\n', env.step(1))

class DNNetwork(nn.Module):
    def __init__(self,layer_size=64):                   # CNN not needed for research internship -> Linear layers, batchnormalisation not needed
        super(DNNetwork,self).__init__()                # super(superclass) - inherit the methods of the superclass (class above this one). Here: inherit all __init__ method of DQN
        self.lin1 = nn.Linear(8,layer_size)             # input (here 8) corresponds to the size of observation space
        self.lin2 = nn.Linear(layer_size,layer_size)
        self.lin3 = nn.Linear(layer_size,4)             # output (here 4) corresponds to the size of action space

        ### For CNNs
        # stide         gives how much the filter is moved across the matrix (e.g. stride = 2 means: move the filter 2 indicies to the right of the matrix)
        # kernel_size   size of the tensor/matrix filter between convolutional layers

    def forward(self,state):
        # x = state.to(torch.device)
        x = tfunc.relu( self.lin1(state) )               # ReLU - rectified linear unit. take max(0,input) of the input
        x = tfunc.relu( self.lin2(x) )
        return tfunc.relu( self.lin3(x) )

### Defining replay memory
transit = namedtuple( 'Transition',('s','a','r','s_next') )

class Replay_memory(object):
    def __init__(self,memory_size, batch_size):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
    def push(self,*args):
        self.memory.append( transit(*args) )
    def sample(self):
        return random.sample(self.memory,self.batch_size)
    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self,):
        self.state_size = 8
        self.action_size = 4
        self.qnet_local = DNNetwork().to(device)
        self.qnet_target = DNNetwork().to(device)
        self.optimize = optim.SGD(self.qnet_local.parameters(), lr=5*10**-4)    # huber loss as alternative?
        self.memory = Replay_memory(100,10)
        self.t_step = 0

    def step(self, *args):
        self.memory.push(*args)
        self.t_step = (self.t_step + 1) % TARGET_UPDATE
        if (self.t_step % TARGET_UPDATE == 0) and (self.memory == 100):
            self.learn( self.memory.sample() )

    def act(self, state, eps=0.):       # copied from tutorial
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, exp, gamma=0.99):
        s, a, r, next_s = exp
        q_target_next = self.qnet_target(next_s).detach().max(1)[0].unsqueeze(1)

        # Bellman equation. Calculating q_target and and current q_value
        q_target = r + gamma * q_target_next * (1-self.t_step)      # q_target
        q_expect = self.qnet_local(s).gather(1, a)                  # current q

        loss = tfunc.mse_loss(q_expect, q_target)   # optimize with mean squared loss
        self.optimize.zero_grad()
        loss.backward()                             # backpropagation
        self.optimize.step()

        self.update(self.qnet_local,self.qnet_target, TAU)

    def update(self, local, target, tau):
        for target, local in zip(target.parameters(), local.parameters()):
            target.data.copy_( tau*local.data + (1-tau)*target.data )

### Training parameters
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1*10**-3
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 5

##### Playground


### Training
def run_agent(runs=2000, gen_life=100):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)   # last 100 scores
    for episode in range(runs):
        state = env.reset()[0]    # if specific seed used, no improvement of the agent
        score = 0
        for i in range(gen_life):
            action = agent.act(state)
            next_state, reward, _, trunc, done = env.step(action)
            agent.step(state, action, reward, next_state)
            state = next_state
            score += reward
            if trunc or done:
                break
        scores_window.append(score)  # save most recent 100 scores
        scores.append(score)

        if episode % 100 == 0:
            print("Running episode %s. Current averaged score: %.2f" % (episode,np.mean(scores_window)))

        if np.mean(scores_window) >= 200.0:
            print("Training done in %s. Average score of 200 or more achieved!" % episode)
            break

    return scores

agent = Agent()
scores = run_agent()

# plot the scores
mplt.plot(np.linspace(0,2000),scores)
mplt.show()

env.close()