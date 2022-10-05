import torch
import torch.nn as nn
import gym

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

# class box():
#     def __init__(self,dimensions):
#         self.dim = dimensions
#
# k = box(24)
# print(k.dim)
#
# m = nn.Softmax(dim=1)
# input = torch.randn(2,3)
# print(input, m(input), sep="\n")

### Environment of the Lunar Lander
# env = gym.make("LunarLander-v2", new_step_api=True)
# obs = env.reset()
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