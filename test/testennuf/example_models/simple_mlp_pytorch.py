#  (C) Crown Copyright, Met Office, 2025.

import torch
import torch.nn as nn

class SimpleMLP:
    @staticmethod
    def build_sequential_simple(device):
        net = nn.Sequential(
            nn.Linear(3, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
            ).to(device)
        return net

    @staticmethod
    def input_shape():
        return (3,)
