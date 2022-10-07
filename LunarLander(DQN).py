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
# obs = env.reset()
# print(obs,'\n', env.step(1))

class DNNetwork(nn.Module):
    def __init__(self,layer_size=64):                   # CNN not needed for research internship -> Linear layers, batchnormalisation not needed
        super(DNNetwork,self).__init__()                # super(superclass) - inherit the methods of the superclass (class above this one). Here: inherit all __init__ method of DQN
        self.layer_size = layer_size
        self.lin1 = nn.Linear(8,layer_size)             # input (here 8) corresponds to the size of observation space
        self.lin2 = nn.Linear(layer_size,layer_size)    # layer_size = amount of neurons between hidden layers
        self.lin3 = nn.Linear(layer_size,4)             # output (here 4) corresponds to the size of action space

        ### For CNNs
        # stide         gives how much the filter is moved across the matrix (e.g. stride = 2 means: move the filter 2 indicies to the right of the matrix)
        # kernel_size   size of the tensor/matrix filter between convolutional layers

    def forward(self,state):
        # x = state.to(torch.device)
        x = tfunc.relu( self.lin1(state) )               # ReLU - rectified linear unit. take max(0,input) of the input
        x = tfunc.relu( self.lin2(x) )
        return self.lin3(x)

### Defining replay memory
memory = namedtuple( 'Memory',('s','a','r','next_s') )

class Replay_memory(object):
    def __init__(self,memory_size, batch_size):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
    def remember(self, *args):                           # *args: put any amount of arguments into the function
        self.memory.append( memory(*args) )
    def get_sample(self):
        return random.sample(self.memory,self.batch_size)
    def batch_size(self):
        return self.batch_size
    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self,memory_size,batch_size):
        self.state_size = 8
        self.action_size = 4
        self.qnet_local = DNNetwork().to(device)
        self.qnet_target = DNNetwork().to(device)
        self.optimize = optim.Adam(self.qnet_local.parameters(), lr=1e-4)    # huber loss as alternative?
        self.memory = Replay_memory(memory_size,batch_size)
        self.batch_size = self.memory.batch_size
        self.t_step = 0

    def evaluate(self, *args):
        self.memory.remember(*args)     # input: s, a, r, next_s
        self.t_step = (self.t_step + 1) % TARGET_UPDATE
        if (self.t_step % TARGET_UPDATE == 0) and ( self.memory.__len__() >= self.batch_size):
            self.learn( self.memory.get_sample() )

    def act(self, input_state, eps=0.):         # parameter eps for eps-greedy action selection
        ### return the best action based on current state and NN
        # convert the array from env into torch.tensor in float form
        state = torch.from_numpy(input_state).float()
        self.qnet_local.eval()                  # set NN in evaluation mode
        # forward current state through the network
        with torch.no_grad():
            action_values = self.qnet_local.forward(state)
        self.qnet_local.train()                 # set NN in training mode
        # return the best action
        return np.argmax( action_values.cpu().data.numpy() )

    def learn(self, exp, gamma=0.9):
        s_tens = torch.randn(10,8)
        a_tens = torch.randn(10).unsqueeze(1).long()
        r_tens = torch.randn(10).unsqueeze(1)
        s_next_tens = torch.rand(10, 8)

        # unpack memories into a tensor/vector with states, actions, or rewards
        for i in range( len(exp) ):
            s_tens[i] = torch.tensor( exp[i].s )
            a_tens[i] = torch.tensor( exp[i].a )
            r_tens[i] = torch.tensor( exp[i].r )
            s_next_tens[i] = torch.tensor( exp[i].next_s )        # attach s_next from each memory

        # Bellman equation. Calculating q_target and and current q_value
        q_target_next = self.qnet_target(s_next_tens).detach()          # get q_values of next states
        q_target = r_tens + gamma * q_target_next                       # q_target
        q_expected = self.qnet_local(s_tens).gather(1, a_tens)          # current q

        loss = tfunc.mse_loss(q_expected, q_target)     # calculate mean squared loss between expected and target q_values
        self.optimize.zero_grad()
        loss.backward()                                 # backpropagation and recalculating the strength of neuron connections in NN
        self.optimize.step()

        self.update(self.qnet_local,self.qnet_target, TAU)

    def update(self, local, target, tau):
        for target, local in zip(target.parameters(), local.parameters()):
            target.data.copy_( tau*local.data + (1-tau)*target.data )

### Training
def run_agent(episodes=2000, play_time=1000):
    print( '| Variables during this run |\n%s\t# of Episodes\n%s\tPlay time\n%s\t\tNN hidden layer size\n%s\t\tTarget update\n%s\tTau'
           '\n%s\t\tAgents memory size\n%s\t\tMemory batch size'
           % (episodes,play_time,LAYER_SIZE,TARGET_UPDATE,TAU,MEMORY_SIZE,BATCH_SIZE) )
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)   # last 100 scores
    for episode in range(episodes):
        state = env.reset()[0]          # if specific seed used, no improvement of the agent
        score = 0
        for time in range(play_time):   # define "playtime" of an agent in environment
            action = agent.act(state)   # act on primary state, get best action from NN
            next_state, reward, terminated, _, done = env.step(action)  # environment takes one step according to chosen the action
            agent.evaluate(state, action, reward, next_state)
            state = next_state
            score += reward
            if terminated or done:
                break
        scores_window.append(score)  # save most recent 100 scores
        scores.append(score)

        if episode % 50 == 0:
            print("Running episode %s. Current averaged score: %.2f" % (episode,np.mean(scores_window)) )

        if np.mean(scores_window) >= 200.0:
            print("Training done in %s. Average score of 200 or more achieved!" % episode)
            break

    return scores

### Training parameters
GAMMA = 0.99
TAU = 1e-3
TARGET_UPDATE = 5
LAYER_SIZE = 64
MEMORY_SIZE = 100
BATCH_SIZE = 10

agent = Agent(memory_size=MEMORY_SIZE,batch_size=BATCH_SIZE)
scores = run_agent()

# plot the scores
# fig = mplt.figure()
# mplt.plot( len(scores), scores)
# mplt.xlabel('Episode #')
# mplt.ylabel('Score')
# mplt.show()

env.close()