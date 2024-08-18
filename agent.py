import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from collections import deque, namedtuple

# defining memory instance
memory = namedtuple('Memory', ('s', 'a', 'r', 'next_s', 'term'))


class DNNetwork(nn.Module):
    def __init__(self, layer_size=64):
        # super(superclass) - inherit the methods of the superclass (class above this one).
        # Here: inherit all __init__ method of DQN
        super(DNNetwork, self).__init__()
        self.layer_size = layer_size
        # CNN not needed for research internship -> Linear layers, batch normalisation not needed
        self.net = nn.Sequential(
            nn.Linear(8, layer_size),   # input (here 8) corresponds to the size of observation space
            nn.ReLU(),                          # ReLU - rectified linear unit. take max(0,input) of the input
            nn.Linear(layer_size, layer_size),  # layer_size = amount of neurons between hidden layers
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 4)    # output (here 4) corresponds to the size of action space
        )

        ### For CNNs
        # stride:    gives how much the filter is moved across the matrix
        # (e.g. stride = 2 means: move the filter 2 indices to the right of the matrix)
        # kernel_size   size of the tensor/matrix filter between convolutional layers

    def forward(self, state):
        return self.net(state)


class ReplayMemory(object):
    def __init__(self, memory_size, batch_size):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def remember(self, *args):  # *args: put any amount of arguments into the function
        self.memory.append(memory(*args))

    def get_sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, hparams):
        self.state_size = 8
        self.action_size = 4
        self.memory_size = hparams['memory_size']
        self.batch_size = hparams['batch_size']
        self.tau = hparams['tau']
        self.gamma = hparams['gamma']
        self.eps = hparams['eps_start']
        self.eps_start = hparams['eps_start']
        self.eps_end = hparams['eps_end']
        self.eps_term = hparams['eps_term']
        self.lr = hparams['lr']
        self.loss = 0
        self.net_update = hparams['net_update']
        self.qnet_local = DNNetwork()
        self.qnet_target = DNNetwork().eval()
        self.optimizer = optim.Adam(self.qnet_local.parameters(), self.lr)    # huber loss as alternative?
        self.memory = ReplayMemory(self.memory_size, self.batch_size)
        self.t_step = 0
        self.s_tens = torch.tensor(np.zeros((self.batch_size, 8))).float()
        self.a_tens = torch.tensor(range(self.batch_size)).unsqueeze(1).long()
        self.r_tens = torch.tensor(range(self.batch_size)).float()
        self.s_next_tens = torch.tensor(np.zeros((self.batch_size, 8))).float()
        self.term_tens = torch.tensor(range(self.batch_size)).long()

    def get_action(self, observation):
        # return the best action based on current state and NN
        state = torch.from_numpy(observation).float()   # convert the array from env into torch.tensor in float form
        with torch.no_grad():
            action_values = self.qnet_local.forward(state)          # forward current state through the network

        # eps-greedy action selection
        if random.random() > self.eps:
            return np.argmax(action_values.detach().numpy())
        else:
            return random.randint(0, 3)

    def evaluate(self, *args):
        self.memory.remember(*args)     # input: s, a, r, next_s, terminated
        self.t_step = self.t_step + 1
        if (self.t_step % self.net_update == 0) and (self.memory.__len__() >= self.batch_size):
            self.learn(self.memory.get_sample())

    def learn(self, exp):
        # unpack memories into a tensor/vector with states, actions, or rewards
        # attach an argument of named tuple from each memory
        for i in range(len(exp)):
            self.s_tens[i] = torch.tensor(exp[i].s)
            self.a_tens[i] = torch.tensor(exp[i].a)
            self.r_tens[i] = torch.tensor(exp[i].r)
            self.s_next_tens[i] = torch.tensor(exp[i].next_s)
            self.term_tens[i] = torch.tensor(exp[i].term)

        # Bellman equation. Calculating q_target and and current q_value
        q_target_next = self.qnet_target(self.s_next_tens)  # get q_values of next states
        q_target = self.r_tens + self.gamma * torch.max(q_target_next, dim=1)[0]*(1-self.term_tens)       # q_target
        q_expected = self.qnet_local(self.s_tens).gather(1, self.a_tens).squeeze()                        # current q

        # optimize the model with backpropagation and no tracing of tensor history
        self.loss = F.mse_loss(q_expected, q_target)    # calculate mean squared loss between expected and target q_values
        self.loss.backward()        # backpropagation and recalculating the strength of neuron connections in NN
        self.optimizer.step()
        self.optimizer.zero_grad()  # zeroing the gradients of the parameters in optimizer

        # update network parameters
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.-self.tau)*target_param.data)

    def get_new_epsilon(self, current_episode):
        """
        Calculates the new :math:`\epsilon` value for each successive :code:`episode`.
        :param current_episode: current training episode
        :return:
        """
        slope = (self.eps_end - self.eps_start) / self.eps_term
        eps = slope * current_episode + self.eps_start
        self.eps = max(self.eps_end, eps)

    def train(self, hparams, environment:gym.Env, episodes: int = 1500, play_time: int = 1000):
        '''
            Initializes the agents' training loop.
            :param hparams:     [dict] containing hyperparameters
            :param episodes:    [int] maximum number of training episodes
            :param play_time:   [int] maximum number of actions per episode
            :return: scores, loss, last_episode
            '''
        # print statement returns currently used variables
        print('| Variables during this run |')
        print(f'\tEpisodes: {episodes}')
        print(f'\tPlay time: {play_time}')
        for k, v in hparams.items():
            print(f'\t{k.upper()}: {v}')

        scores, loss = [], []
        last_scores, last_loss = deque(maxlen=100), deque(maxlen=100)

        for episode in range(episodes):
            state = environment.reset()[0]
            score = 0
            for time in range(play_time):
                # act on primary state and get best action from NN
                action = self.get_action(state)
                # take a step in the environment according to the chosen action
                next_state, reward, terminated, truncated, _ = environment.step(action)
                self.evaluate(state, action, reward, next_state, terminated)
                state = next_state
                score += reward
                if terminated or truncated:
                    break
                self.get_new_epsilon(current_episode=episode)

            scores.append(score)
            loss.append(int(self.loss))
            last_scores.append(score)
            last_loss.append(int(self.loss))

            if episode % 50 == 0:
                print(f'Running episode {episode}. Currently averaged score: {np.mean(last_scores)}')
                print(f'Loss average: {np.mean(last_loss)}')
                if self.eps > self.eps_end:
                    print(f'Epsilon: {self.eps}')

            if np.mean(last_scores) >= 200.0:
                print(f'Environment solved! Training done in {episode} episodes.')
                print(f'Average loss: {np.mean(last_loss)}')
                torch.save(self.qnet_local.state_dict(), f'./diagnostics/state_dicts/state_dict.pt')
                environment.close()
                break

        return scores, loss, episode
