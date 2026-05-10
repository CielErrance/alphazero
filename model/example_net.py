import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

from env.base_env import BaseGame

class BaseNetConfig:
    def __init__(
        self, 
        num_channels:int = 256,
        dropout:float = 0.3,
        linear_hidden:list[int] = [256, 128],
        num_res_blocks:int = 2,
    ):
        self.num_channels = num_channels
        self.linear_hidden = linear_hidden
        self.dropout = dropout
        self.num_res_blocks = num_res_blocks
        
class MLPNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        input_dim = observation_size[0] * observation_size[1] if len(observation_size) == 2 else observation_size[0]
        self.layer1 = nn.Linear(input_dim, config.linear_hidden[0])
        self.layer2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, x: torch.Tensor):
        #                                                         x: batch_size x board_x x board_y
        x = x.view(x.size(0), -1) # reshape tensor to 1d vectors, x.size(0) is batch size
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        pi = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class LinearModel(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super(LinearModel, self).__init__()
        
        self.action_size = action_space_size
        self.config = config
        self.device = device
        
        observation_size = reduce(lambda x, y: x*y , observation_size, 1)
        self.l_pi = nn.Linear(observation_size, action_space_size)
        self.l_v  = nn.Linear(observation_size, 1)
        self.to(device)
    
    def forward(self, s: torch.Tensor):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(s.shape[0], -1)                                # s: batch_size x (board_x * board_y)
        pi = self.l_pi(s)
        v = self.l_v(s)
        return F.log_softmax(pi, dim=1), torch.tanh(v)
    
class MyNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        if len(observation_size) != 2:
            raise ValueError("MyNet expects 2D board observations")
        self.board_x, self.board_y = observation_size
        self.action_size = action_space_size

        self.conv_in = nn.Conv2d(1, config.num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(config.num_channels)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(config.num_channels) for _ in range(config.num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(config.num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, action_space_size)

        self.value_conv = nn.Conv2d(config.num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_x * self.board_y, config.linear_hidden[0])
        self.value_fc2 = nn.Linear(config.linear_hidden[0], 1)

        self.to(device)

    
    def forward(self, s: torch.Tensor):
        ########################
        # TODO: your code here #
        ########################
        x = s.view(-1, 1, self.board_x, self.board_y)
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        pi = self.policy_conv(x)
        pi = F.relu(self.policy_bn(pi))
        pi = pi.view(pi.size(0), -1)
        pi = self.policy_fc(pi)

        v = self.value_conv(x)
        v = F.relu(self.value_bn(v))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)

        return F.log_softmax(pi, dim=1), torch.tanh(v)



class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out