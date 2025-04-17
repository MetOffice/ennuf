#  (C) Crown Copyright, Met Office, 2025.

import torch
import torch.nn as nn

class SimpleMLP:
    def build_sequential_simple:
        net = nn.Sequential(
            nn.Linear(3, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
            ).to(device)
        return net