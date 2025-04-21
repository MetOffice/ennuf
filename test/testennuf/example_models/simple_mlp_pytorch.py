#  (C) Crown Copyright, Met Office, 2025.

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleMLP:
    def build_sequential_simple():
        net = nn.Sequential(
            nn.Linear(1, 6),
            nn.Sigmoid(),
            nn.Linear(6, 4),
            nn.ReLU()
            ).to(device)
        return net