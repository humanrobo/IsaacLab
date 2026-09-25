import torch
import torch.nn as nn

class RNDNetwork(nn.Module):
    def __init__(self, input_dim=10, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x):
        return self.net(x)

class RandomNetworkDistillation(nn.Module):
    def __init__(self, input_dim=10, feature_dim=128, learning_rate=1e-4, device="cuda"):
        super().__init__()
        self.target = RNDNetwork(input_dim, feature_dim).to(device)
        self.predictor = RNDNetwork(input_dim, feature_dim).to(device)
        for param in self.target.parameters():
            param.requires_grad = False
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=learning_rate,
        )

    @torch.no_grad()
    def get_intrinsic_reward(self, observations):
        target = self.target(observations)
        prediction = self.predictor(observations)
        return torch.mean((target - prediction) ** 2, dim=-1, keepdim=True)

    def update(self, observations):
        with torch.no_grad():
            target = self.target(observations)
        prediction = self.predictor(observations)
        loss = torch.mean((prediction - target) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()