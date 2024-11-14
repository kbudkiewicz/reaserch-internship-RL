from agent import *
from configs import AgentConfig

# assign the device and initialize the agent
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Current device: {device.upper()}')
agent = Agent(AgentConfig)

# initializing the Lunar Lander environment
env = gym.make("LunarLander-v2")

# visualize: uncomment code below to run the best agent trained so far
# render_mode='human' for visual render after the training is done
# hparams['eps_start'] = 0
# env = gym.make("LunarLander-v2", render_mode='human')

# WARNING: if network variables are named differently in the dictionary and code then an error occurs
# -> same variable names so import can work properly
# agent.qnet_local.load_state_dict(torch.load('diagnostics/state_dicts/state_dict_.pt'))
# agent.qnet_local.eval()
# agent.qnet_target.load_state_dict(torch.load('diagnostics/state_dicts/state_dict_.pt'))

scores, loss, last_episode = agent.train(environment=env)
print(f'Scores: {scores}\n'
      f'Loss: {loss}\n')
