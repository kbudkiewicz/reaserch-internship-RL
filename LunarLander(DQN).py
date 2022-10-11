import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as mpl
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
env = gym.make("LunarLander-v2", render='human')

class DNNetwork(nn.Module):
    def __init__(self,layer_size=64):                   # CNN not needed for research internship -> Linear layers, batchnormalisation not needed
        super(DNNetwork,self).__init__()                # super(superclass) - inherit the methods of the superclass (class above this one). Here: inherit all __init__ method of DQN
        self.layer_size = layer_size
        self.lin1 = nn.Linear(8,layer_size)             # input (here 8) corresponds to the size of observation space
        self.lin2 = nn.Linear(layer_size,layer_size)    # layer_size = amount of neurons between hidden layers
        self.lin3 = nn.Linear(layer_size,4)             # output (here 4) corresponds to the size of action space
        self.to(device)

        ### For CNNs
        # stide         gives how much the filter is moved across the matrix (e.g. stride = 2 means: move the filter 2 indicies to the right of the matrix)
        # kernel_size   size of the tensor/matrix filter between convolutional layers

    def forward(self,state):
        x = F.relu( self.lin1(state) )               # ReLU - rectified linear unit. take max(0,input) of the input
        x = F.relu( self.lin2(x) )
        action_set = self.lin3(x)
        return action_set

### Defining replay memory
memory = namedtuple( 'Memory', ('s','a','r','next_s','term') )

class Replay_memory(object):
    def __init__(self,memory_size, batch_size):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
    def remember(self, *args):              # *args: put any amount of arguments into the function
        self.memory.append( memory(*args) )
    def get_sample(self):
        return random.sample(self.memory,self.batch_size)
    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, memory_size, batch_size, tau, gamma, epsilon, learning_rate):
        self.state_size = 8
        self.action_size = 4
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.eps = epsilon
        self.lr = learning_rate
        self.loss = 300
        self.qnet_local = DNNetwork().to(device)
        self.qnet_target = DNNetwork().to(device).eval()
        self.optimizer = optim.Adam(self.qnet_local.parameters(), self.lr)    # huber loss as alternative?
        self.memory = Replay_memory(memory_size,batch_size)
        self.t_step = 0

        l = []
        for i in range(self.batch_size):
            l.append(i)
        self.list_size = l

    def get_action(self, observation):
        # return the best action based on current state and NN
        state = torch.from_numpy(observation).float()               # convert the array from env into torch.tensor in float form
        with torch.no_grad():
            action_values = self.qnet_local.forward(state)          # forward current state through the network. Try with no_grad

        # eps-greedy action selection
        if random.random() > self.eps:
            return np.argmax( action_values.detach().numpy() )
        else:
            return random.randint(0,3)      # take random action out of 4

    def evaluate(self, *args):
        self.memory.remember(*args)     # input: s, a, r, next_s, terminated
        self.t_step = self.t_step + 1
        if (self.t_step % NET_UPDATE == 0) and ( self.memory.__len__() >= self.batch_size ):
            self.learn( self.memory.get_sample() )

    def learn(self, exp):
        s_tens = torch.tensor( np.zeros((self.batch_size,8)) ).float()
        a_tens = torch.tensor( self.list_size ).unsqueeze(1).long()
        r_tens = torch.tensor( self.list_size ).float()
        s_next_tens = torch.tensor( np.zeros((self.batch_size,8)) ).float()
        term_tens = torch.tensor( self.list_size ).unsqueeze(1).long()

        # unpack memories into a tensor/vector with states, actions, or rewards
        for i in range( len(exp) ):
            s_tens[i] = torch.tensor( exp[i].s )
            a_tens[i] = torch.tensor( exp[i].a )
            r_tens[i] = torch.tensor( exp[i].r )
            s_next_tens[i] = torch.tensor( exp[i].next_s )      # attach s_next from each memory
            term_tens[i] = torch.tensor( exp[i].term )

        # Bellman equation. Calculating q_target and and current q_value
        q_target_next = self.qnet_target(s_next_tens)                                           # get q_values of next states
        q_target = r_tens + self.gamma * torch.max(q_target_next, dim=1)[0]#*(1-term_tens)      # q_target
        q_expected = self.qnet_local(s_tens).gather(1, a_tens).squeeze()                        # current q

        # optimize the model with backpropagation and no tracing of tensor history
        self.loss = F.mse_loss(q_expected, q_target)        # calculate mean squared loss between expected and target q_values
        self.loss.backward()                                # backpropagation and recalculating the strength of neuron connections in NN
        self.optimizer.step()
        self.optimizer.zero_grad()  # zeroing the gradients of the parameters in optimizer

        # update network parameters
        for target_param, local_param in zip( self.qnet_target.parameters(), self.qnet_local.parameters() ):
            target_param.data.copy_( self.tau*local_param.data + (1.-self.tau)*target_param.data )

### Training
def run_agent(episodes=3000, play_time=1000):
    # print statement returns currently used variables
    print( '| Variables during this run |\n'+ 60*'-' + '\n%s\t\t# of Episodes\n%s\t\tPlay time\n%s\t\t\tNN\' hidden layer size\n%s\t\t\tNetwork update'
           '\n%s\t\tAgents memory size\n%s\t\t\tMemory batch size\n%s\t\tTau\n%s\t\tGamma\n%s\t\tLearning rate\n'
           % (episodes,play_time,LAYER_SIZE,NET_UPDATE,MEMORY_SIZE,BATCH_SIZE,TAU,GAMMA,LR)  + 60*'-' )

    scores = []
    last_scores, loss = deque(maxlen=100), deque(maxlen=50)
    for episode in range(episodes):
        state = env.reset()[0]
        score = 0
        for time in range(play_time):                                           # "playtime" = max amount of steps allowed in environment
            action = agent.get_action(state)                                    # act on primary state, get best action from NN
            next_state, reward, terminated, truncated, _ = env.step(action)     # environment takes one step according to chosen the action
            agent.evaluate(state, action, reward, next_state, terminated)
            state = next_state
            score += reward
            if terminated or truncated:
                break
        agent.eps = max(EPS_END,EPS_DEC*agent.eps)  # update eps
        last_scores.append(score)                   # save most recent 100 scores
        scores.append(score)
        loss.append( int(agent.loss) )

        if episode % 50 == 0:
            print("Running episode %s. Currently averaged score: %.2f" % (episode, np.mean(last_scores)) )
            print('Loss average = %s' % np.mean(loss))

            # diagnostics
            # mpl.figure(1)   # scores
            # x = np.linspace(0,episode, len(scores))
            # y = scores
            # mpl.plot(x,y)
            # mpl.figure(2)   # loss
            # x = np.linspace(0, episode, len(loss))
            # y = loss
            # mpl.plot(x, y)

        if np.mean(last_scores) >= 200.0:
            print("Environment solved! Training done in %s episodes." % episode)
            break

    return scores

### Training parameters
GAMMA = 0.99
TAU = 1e-3
NET_UPDATE = 10
LAYER_SIZE = 64
MEMORY_SIZE = 50000
BATCH_SIZE = 100
LR = 1e-4
EPS = 1.0
EPS_END = 1e-2
EPS_DEC = 0.995

agent = Agent(memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, learning_rate=LR, epsilon=EPS)
scores = run_agent()
env.close()

# plot the scores
# fig = mplt.figure()
# mplt.plot( len(scores), scores)
# mplt.xlabel('Episode #')
# mplt.ylabel('Score')
# mplt.show()