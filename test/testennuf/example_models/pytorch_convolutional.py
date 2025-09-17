# python
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PytorchConvolutional:
    @staticmethod
    def build_only_conv():
        model = nn.Sequential(
            nn.Conv1d(3,4,kernel_size=3,padding=1),
        )
        return model.to(device)

    @staticmethod
    def build_simple_conv():
        """Simple 1D ConvNet with one conv and pooling layer"""
        model = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(16 * 64, 10)  # input length 128 -> pooled to 64
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
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 5)
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

    @staticmethod
    def build_conv_multi_output():
        """1D ConvNet with multiple outputs (returns a custom nn.Module)"""
        class MultiOutputNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(3, 16, kernel_size=3, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool1d(2)  # 128 -> 64
                self.flatten = nn.Flatten()
                self.fc_class = nn.Linear(16 * 64, 4)
                self.fc_score = nn.Linear(16 * 64, 1)

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                x = self.pool(x)
                x = self.flatten(x)
                out1 = self.fc_class(x)
                out2 = torch.sigmoid(self.fc_score(x))
                return {"class_output": out1, "score_output": out2}

        return MultiOutputNet().to(device)