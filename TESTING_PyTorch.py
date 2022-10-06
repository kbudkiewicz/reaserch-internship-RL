import torch
import torch.nn as nn
import torch.nn.functional as tfunc
import gym

### This file is for testing/learning pytorch and gym functions and how to apply them correctly

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

# m = nn.Softmax(dim=1)
# input = torch.randn(2,3)
# print(input, m(input), sep="\n")

### Environment of the Lunar Lander
# env = gym.make("LunarLander-v2")
# obs = env.reset(seed=0)
# print(env.step(0),"\nLength: ", len(env.step(0)))
# print(obs, "\n")
# print(env.action_space)
# action = env.action_space.sample()
# print(action, "\n")
# print(env.observation_space)
# print(env.step(0))

### squeeze(), flatten() as opposites
# x = torch.tensor([1,2,3,4])
# x = torch.unsqueeze(x,1)
# x = torch.unsqueeze(x,0)
# print(x)
# x = torch.flatten(x)
# x = torch.flatten(x)
# print(x)

### handling of PyTroch NN
# N1 = nn.Linear(8, 32)
# N2 = nn.Linear(32,4)
# input = torch.from_numpy( gym.make('LunarLander-v2').reset()[0] )
# print('State from reset environment:\n', input)
# output = N1(input)
# print('\nState run through NN:\n', output)
# output = N2(output)
# print(output)

### torch.from_numpy() - convert a numpy array into a torch.tensor()

# net = nn.Linear(6, 30)
# net2 = nn.Linear(30,2)
# T = torch.randn(2, 6)
# T2 = torch.tensor([[1,2,3,4,5,6],[-1,2,-3,4,-5,6]])
# print('\nTensor with ReLU applied:')
# T = tfunc.relu(T)
# T2 = tfunc.relu(T2)
# print(T, '\n',T2)
# print('\nTensor goes through NN:')
# print( net(T) )
# print( net2( net(T) ) )
# # print( net(T2) )
# print('\nReLU applied to NN:')
# print( tfunc.relu( net(T) ) )

##### Gym
# env = gym.make("LunarLander-v2", render_mode="human")
#
# observation, info = env.reset(seed=0)
# print(observation, "\n", info)
#
# for runs in range(200):
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     # print("Obs: %s\nr = %s\nTerm: %s\nTrunc: %s\nInfo: %s"%(observation,reward,terminated,truncated,info))
#
#     if terminated:
#         observation, info = env.reset()
#         env.close()