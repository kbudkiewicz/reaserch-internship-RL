class AgentConfig:
    state_space: int = 8
    action_space: int = 4
    memory_size: int = 100_000
    episodes: int = 1_000
    play_time: int = 1_000
    loss: float = 0
    t_step: int = 0
    batch_size: int = 64
    tau: float = 0.01
    gamma: float = 0.99
    eps: float = 1.0
    eps_start: float = 0.01
    eps_end: float = 0.99
    lr: float = 0.001
    net_update_freq: int = 6
