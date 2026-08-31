import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.heightmap_features=nn.Sequential(
            nn.Conv2d(1,16,5,stride=2,padding=0),
            nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=0),
            nn.ReLU(),
            nn.Conv2d(32,32,3,stride=2,padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.net=nn.Sequential(
            nn.Linear(2059,256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU(),
        )
        self.action_mean=nn.Linear(128,2)
        self.log_std=nn.Parameter(torch.full((2,),-1.0))

    def forward(self,ray_heightmap,policy_obs):
        x=self.heightmap_features(ray_heightmap)
        x=torch.cat([x,policy_obs],dim=-1)
        x=self.net(x)
        mean=self.action_mean(x)
        std=torch.exp(self.log_std)
        return mean,std