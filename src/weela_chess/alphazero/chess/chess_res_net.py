import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from weela_chess.chess_utils.all_possible_moves import categorical_idx_to_uci_codes


class ChessResNetDNA(BaseModel):
    num_blocks: int
    num_channels: int

    num_input_channels: int


class ChessResBlock(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ChessResNet(nn.Module):
    def __init__(self, genome: ChessResNetDNA, device: torch.device):
        super().__init__()
        n_actions = len(categorical_idx_to_uci_codes())
        num_blocks, num_channels, num_input_channels = tuple(genome.model_dump().values())

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(num_input_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ChessResBlock(num_channels) for i in range(num_blocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, n_actions)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_channels, num_input_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_input_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_input_channels * 8 * 8, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
