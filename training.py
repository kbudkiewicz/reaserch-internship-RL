from agent import *

# Assigning the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Current device: {device.upper()}')

# initializing gym's Lunar Lander environment
env = gym.make("LunarLander-v2")

# training
# initialize the training hyperparameters
hparams = {'gamma': 0.99,
           'tau': 2.5e-3,
           'net_update': 6,
           'lr': 1e-3,
           'batch_size': 100,
           'memory_size': 100_000,
           'eps_start': 0.7,
           'eps_end': 0.05,
           'eps_term': 500}    # episode No at which EPS_END will be achieved

# visualize: uncomment code below to run the best agent trained so far
# render_mode='human' for visual render after the training is done
# hparams['eps_start'] = 0
env = gym.make("LunarLander-v2")#, render_mode='human')

# initialize the agent
agent = Agent(hparams=hparams)

# WARNING: if network variables are named differently in the dictionary and code then an error occurs
# -> same variable names so import can work properly
# agent.qnet_local.load_state_dict(torch.load('diagnostics/state_dicts/state_dict_.pt'))
# agent.qnet_local.eval()
# agent.qnet_target.load_state_dict(torch.load('diagnostics/state_dicts/state_dict_.pt'))

scores, loss, last_episode = agent.train(hparams, environment=env)
print(f'Scores: {scores}\n'
      f'Loss: {loss}\n')
