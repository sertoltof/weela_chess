import torch
import torch.nn as nn
import torch.nn.functional as F
from weela_chess.chess_utils.all_possible_moves import all_uci_codes_to_moves, categorical_idx_to_uci_codes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(8, 64, 3, 1)
        # self.conv2 = nn.Conv2d(64, 128, 3, 1)
        # self.conv3 = nn.Conv2d(128, 128, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.0),
            nn.Dropout(0.0),
            nn.Dropout(0.0),
            nn.Dropout(0.15)
        ])

        self.linears = nn.ModuleList([
            nn.Linear(768, 768),
            nn.Linear(768, 768),
            nn.Linear(768, 256),
            nn.Linear(256, len(categorical_idx_to_uci_codes()))
        ])

        # self.fc1 = nn.Linear(1536, 1536)
        # self.fc2 = nn.Linear(1536, 256)
        # self.fc3 = nn.Linear(256, len(categorical_idx_to_uci_codes()))

    def forward(self, x):
        x = torch.flatten(x, 1)
        for linear, drop in zip(self.linears, self.dropouts):
            x = linear(x)
            x = F.relu(x)
            x = drop(x)

        output = F.log_softmax(x, dim=1)
        return output