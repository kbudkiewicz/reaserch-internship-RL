import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, *dims,
                 activation: nn.Module = nn.ReLU(),
                 regularizer: nn.Module = nn.Dropout(p=0.1)):
        super().__init__()
        self.dims = dims
        self.module_list = nn.ModuleList()
        self.activation = activation
        self.regularizer = regularizer

        for idx in range(len(self.dims) - 2):
            self.module_list.append(nn.Linear(self.dims[idx], self.dims[idx + 1]))
            self.module_list.append(self.activation)
            self.module_list.append(self.regularizer)
        self.module_list.append(nn.Linear(dims[-2], dims[-1]))  # last layer without activation
        self.module_list.append(self.regularizer)

        self.net = nn.Sequential(*self.module_list)

    def forward(self, state):
        return self.net(state)