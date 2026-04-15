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

class SimpleNet(torch.nn.Module):
    """PyTorch module multiplying an input vector by 2."""

    def __init__(self) -> None:
        """Initialize the SimpleNet model with predefined weights."""
        super().__init__()
        self._fwd_seq = torch.nn.Sequential(
            torch.nn.Linear(5, 5, bias=False),
        )
        with torch.inference_mode():
            self._fwd_seq[0].weight = torch.nn.Parameter(2.0 * torch.eye(5))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Pass ``batch`` through the model.

        Parameters
        ----------
        batch : torch.Tensor
            A mini-batch of input vectors of length 5.

        Returns
        -------
        torch.Tensor
            batch scaled by 2.
        """
        return self._fwd_seq(batch)
