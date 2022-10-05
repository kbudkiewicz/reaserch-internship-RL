import torch
import torch.nn as nn
import gym

class box():
    def __init__(self,dimensions):
        self.dim = dimensions

k = box(24)
print(k.dim)

m = nn.Softmax(dim=1)
input = torch.randn(2,3)
print(input, m(input), sep="\n")

### Environment of the Lunar Lander
env = gym.make("LunarLander-v2", new_step_api=True)
obs = env.reset()
print(obs, "\n")
print(env.action_space)
action = env.action_space.sample()
print(action, "\n")
print(env.observation_space)
print(env.step(0))