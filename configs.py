class AgentConfig:
    '''
    Hyperparameters for the RL agent. Contains both agent, as well as network hyperparameters.
    '''
    state_space: int = 8
    action_space: int = 4
    memory_size: int = 100_000
    episodes: int = 1_500
    play_time: int = 1_000
    t_step: int = 0
    batch_size: int = 64
    tau: float = 2.5e-3     # soft parameter update constant
    gamma: float = 0.99     # discount factor
    lr: float = 1e-3
    net_update_freq: int = 6
    eps_start: float = 0.7  # starting epsilon value
    eps_end: float = 0.05   # final epsilon value
    eps_term: int = 500     # episode # at which eps_end is reached4
