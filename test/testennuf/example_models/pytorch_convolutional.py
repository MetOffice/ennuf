#  (C) Crown Copyright, Met Office, 2025.
# python
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PytorchConvolutional:
    @staticmethod
    def only_flatten():
        model = nn.Sequential(
            nn.Flatten(0),
        )
        return model.to(device)

    @staticmethod
    def build_only_conv():
        model = nn.Sequential(
            nn.Conv1d(3,5,kernel_size=3,padding=0),
        )
        return model.to(device)
    @staticmethod
    def build_only_conv_no_bias():
        model = nn.Sequential(
            nn.Conv1d(3,4,kernel_size=3,padding=1, bias=False),
        )
        return model.to(device)

    @staticmethod
    def build_simple_conv():
        """Simple 1D ConvNet with one conv and pooling layer"""
        model = nn.Sequential(
            nn.Conv1d(3, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(0),
            nn.Linear(4, 5)
        )
        return model.to(device)

    @staticmethod
    def build_deep_conv():
        """Deeper 1D ConvNet with multiple conv and pooling layers"""
        model = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256 -> 128
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 -> 64
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(1),
            nn.Flatten(0),
            nn.Linear(1024, 5)
        )
        return model.to(device)

    @staticmethod
    def build_conv_with_dropout():
        """1D ConvNet with dropout for regularization"""
        model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 100 -> 49
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32 * 49, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
        return model.to(device)
