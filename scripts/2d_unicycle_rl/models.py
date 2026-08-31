import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

class Policy(GaussianMixin,Model):
    def __init__(self,observation_space,state_space,action_space,device,clip_actions=False,clip_mean_actions=False,clip_log_std=True,min_log_std=-20.0,max_log_std=2.0,reduction="sum"):
        Model.__init__(self,observation_space=observation_space,state_space=state_space,action_space=action_space,device=device)
        GaussianMixin.__init__(self,clip_actions=clip_actions,clip_mean_actions=clip_mean_actions,clip_log_std=clip_log_std,min_log_std=min_log_std,max_log_std=max_log_std,reduction=reduction)
        self.heightmap_features=nn.Sequential(
            nn.Conv2d(1,16,5,stride=2,padding=0),
            nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=0),
            nn.ReLU(),
            nn.Conv2d(32,32,3,stride=2,padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.net=nn.Sequential(
            nn.Linear(2059,256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU()
        )
        self.mean_layer=nn.Linear(128,self.num_actions)
        self.log_std_parameter=nn.Parameter(torch.full((self.num_actions,),-1.0))
    def compute(self,inputs,role):
        observations=inputs["observations"]
        heightmap=observations[:,:6400].reshape(-1,1,80,80)
        policy_obs=observations[:,6400:6411]
        x=self.heightmap_features(heightmap)
        x=torch.cat([x,policy_obs],dim=-1)
        x=self.net(x)
        mean=self.mean_layer(x)
        return mean,{"log_std":self.log_std_parameter}

class Value(DeterministicMixin,Model):
    def __init__(self,observation_space,state_space,action_space,device,clip_actions=False):
        Model.__init__(self,observation_space=observation_space,state_space=state_space,action_space=action_space,device=device)
        DeterministicMixin.__init__(self,clip_actions=clip_actions)
        self.heightmap_features=nn.Sequential(
            nn.Conv2d(1,16,5,stride=2,padding=0),
            nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=0),
            nn.ReLU(),
            nn.Conv2d(32,32,3,stride=2,padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.net=nn.Sequential(
            nn.Linear(2059,256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU()
        )
        self.value_layer=nn.Linear(128,1)
    def compute(self,inputs,role):
        observations=inputs["observations"]
        heightmap=observations[:,:6400].reshape(-1,1,80,80)
        policy_obs=observations[:,6400:6411]
        x=self.heightmap_features(heightmap)
        x=torch.cat([x,policy_obs],dim=-1)
        x=self.net(x)
        return self.value_layer(x),{}