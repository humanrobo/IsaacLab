import torch
import torch.nn as nn

class HeightMapCNN(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        # 80 -> 40 -> 20 -> 10
        # 最後は 64 * 10 * 10 = 6400
        self.fc = nn.Sequential(
            nn.Linear(64 * 10 * 10, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x