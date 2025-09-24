#  (C) Crown Copyright, Met Office, 2025.

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleMLP:
    @staticmethod
    def build_sequential_simple():
        net = nn.Sequential(
            nn.Linear(1, 6),
            nn.Sigmoid(),
            nn.Linear(6, 4),
            nn.ReLU()
            ).to(device)
        return net

class LessSimpleMLP:
    @staticmethod
    def build_sequential():
        net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Sigmoid(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20,2),
            nn.LeakyReLU(negative_slope=0.02)
            ).to(device)
        return net
