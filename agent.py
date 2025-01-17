import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from collections import deque, namedtuple
from diagnostics.diagnostics import diagnose

### Assigning the device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Current device: %s \n" % device.upper())

### Creating gym' Lunar Lander environment
env = gym.make("LunarLander-v2", render_mode='human')    # render_mode='human'
`
class DNNetwork(nn.Module):
    def __init__(self,layer_size=64):                   # CNN not needed for research internship -> Linear layers, batchnormalisation not needed
        super(DNNetwork,self).__init__()                # super(superclass) - inherit the methods of the superclass (class above this one). Here: inherit all __init__ method of DQN
        self.layer_size = layer_size
        self.lin1 = nn.Linear(8,layer_size)             # input (here 8) corresponds to the size of observation space
        self.lin2 = nn.Linear(layer_size,layer_size)    # layer_size = amount of neurons between hidden layers
        self.lin3 = nn.Linear(layer_size,layer_size)
        self.lin4 = nn.Linear(layer_size,4)             # output (here 4) corresponds to the size of action space

        ### For CNNs
        # stide         gives how much the filter is moved across the matrix (e.g. stride = 2 means: move the filter 2 indicies to the right of the matrix)
        # kernel_size   size of the tensor/matrix filter between convolutional layers

    def forward(self,state):
        x = F.relu( self.lin1(state) )               # ReLU - rectified linear unit. take max(0,input) of the input
        x = F.relu( self.lin2(x) )
        x = F.relu( self.lin3(x) )
        action_set = self.lin4(x)
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
        self.qnet_local = DNNetwork()
        self.qnet_target = DNNetwork().eval()
        self.optimizer = optim.Adam(self.qnet_local.parameters(), self.lr)    # huber loss as alternative?
        self.memory = Replay_memory(memory_size,batch_size)
        self.t_step = 0

        self.list_size = [i for i in range(self.batch_size)]
        self.s_tens = torch.tensor( np.zeros((self.batch_size, 8)) ).float()
        self.a_tens = torch.tensor(self.list_size).unsqueeze(1).long()
        self.r_tens = torch.tensor(self.list_size).float()
        self.s_next_tens = torch.tensor( np.zeros((self.batch_size, 8)) ).float()
        self.term_tens = torch.tensor(self.list_size).long()

    def get_action(self, observation):
        # return the best action based on current state and NN
        state = torch.from_numpy(observation).float()               # convert the array from env into torch.tensor in float form
        with torch.no_grad():
            action_values = self.qnet_local.forward(state)          # forward current state through the network

        # eps-greedy action selection
        if random.random() > self.eps:
            return np.argmax( action_values.detach().numpy() )
        else:
            return random.randint(0,3)

    def evaluate(self, *args):
        self.memory.remember(*args)     # input: s, a, r, next_s, terminated
        self.t_step = self.t_step + 1
        if (self.t_step % NET_UPDATE == 0) and ( self.memory.__len__() >= self.batch_size ):
            self.learn( self.memory.get_sample() )

    def learn(self, exp):
        # unpack memories into a tensor/vector with states, actions, or rewards. attach an argument of named tuple from each memory
        for i in range( len(exp) ):
            self.s_tens[i] = torch.tensor( exp[i].s )
            self.a_tens[i] = torch.tensor( exp[i].a )
            self.r_tens[i] = torch.tensor( exp[i].r )
            self.s_next_tens[i] = torch.tensor( exp[i].next_s )
            self.term_tens[i] = torch.tensor( exp[i].term )

        # Bellman equation. Calculating q_target and and current q_value
        q_target_next = self.qnet_target(self.s_next_tens)                                                # get q_values of next states
        q_target = self.r_tens + self.gamma * torch.max(q_target_next, dim=1)[0]*(1-self.term_tens)       # q_target
        q_expected = self.qnet_local(self.s_tens).gather(1, self.a_tens).squeeze()                        # current q

        # optimize the model with backpropagation and no tracing of tensor history
        self.loss = F.mse_loss(q_expected, q_target)        # calculate mean squared loss between expected and target q_values
        self.loss.backward()                                # backpropagation and recalculating the strength of neuron connections in NN
        self.optimizer.step()
        self.optimizer.zero_grad()                          # zeroing the gradients of the parameters in optimizer

        # update network parameters
        for target_param, local_param in zip( self.qnet_target.parameters(), self.qnet_local.parameters() ):
            target_param.data.copy_( self.tau*local_param.data + (1.-self.tau)*target_param.data )

def calc_eps(current_episode, eps_start, eps_end, eps_term):
    # calculation of epsilon based on a linear function
    slope = (eps_end - eps_start) / eps_term
    eps = slope * current_episode + eps_start
    return eps

### Training
def run_agent(episodes=1500, play_time=1000):
    # print statement returns currently used variables
    print(  '| Variables during this run |\n'+ 60*'-' +
            '\n%s\t\t# of Episodes\n'
            '%s\t\tPlay time\n'
            '%s\t\t\tNN\' hidden layer size\n'
            '%s\t\t\tNetwork update\n'
            '%s\t\tAgents memory size\n'
            '%s\t\t\tMemory batch size\n'
            '%s\t\tTau\n'
            '%s\t\tGamma\n'
            '%s\t\tLearning rate\n'
            '%s\t\t\tEpsilon start\n'
            '%s\t\tEpsilon end\n'
            '%s\t\tEpsilon termination\n'
            % (episodes,play_time,LAYER_SIZE,NET_UPDATE,MEMORY_SIZE,BATCH_SIZE,TAU,GAMMA,LR,EPS_START,EPS_END,EPS_TERM)
            + 60*'-' )

    scores, loss = [], []
    last_scores, last_loss = deque(maxlen=100), deque(maxlen=100)
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
        agent.eps = max( EPS_END, calc_eps(episode,EPS_START,EPS_END,EPS_TERM) )         # recalculate current eps linearly

        scores.append(score)
        loss.append( int(agent.loss) )
        last_scores.append(score)
        last_loss.append( int(agent.loss) )

        if episode % 50 == 0:
            print( "Running episode %s. Currently averaged score: %.2f" % (episode, np.mean(last_scores)) )
            print( 'Loss average = %s' % np.mean(last_loss) )
            if agent.eps > EPS_END:
                print('Eps = %s' % agent.eps)

        if np.mean(last_scores) >= 200.0:
            print("\nEnvironment solved! Training done in %s episodes." % episode)
            print( 'Loss average = %s' % np.mean(last_loss) )
            # torch.save(agent.qnet_local.state_dict(), 'Diagnostics/Linear epsilon decay/state_dict1.pt')
            env.close()
            break

    return scores, loss, episode

### Training parameters
GAMMA = 0.99
TAU = 2.5e-3
NET_UPDATE = 6
LAYER_SIZE = 64
MEMORY_SIZE = 100000
BATCH_SIZE = 100
LR = 1e-3
EPS_START = 0.7
EPS_END = 0.05
EPS_TERM = 1000        # value at which EPS_END will be achieved

### importing of state dictionary of already trained agent
EPS_START, EPS_END = 0, 0
agent = Agent(memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU, learning_rate=LR, epsilon=EPS_START)
agent.qnet_local.load_state_dict( torch.load('C:/Users/kryst/Documents/GitHub/research-internship-RL/Diagnostics/Exponential epsilon decay/Using state dictionary from previous runs/state_dict.pt') )
agent.qnet_local.eval()
agent.qnet_target.load_state_dict( torch.load('C:/Users/kryst/Documents/GitHub/research-internship-RL/Diagnostics/Exponential epsilon decay/Using state dictionary from previous runs/state_dict.pt'))
# if network variables are named differently in the dictionary and code then an error occurs -> same variable names so import can work properly

scores, loss, last_episode = run_agent()
print(scores)
print(loss)
# diagnose(scores, loss, last_episode)