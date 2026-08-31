import torch
from skrl_env import Unicycle2DSkrlEnv
from models import Policy
env=Unicycle2DSkrlEnv(num_envs=4096,device="cuda")
policy=Policy().cuda()
obs=env.reset()
with torch.no_grad():
    mean,std=policy(obs["ray_heightmap"],obs["policy_obs"])
print("mean:",mean.shape)
print("std:",std.shape)
print("mean:",mean.mean().item())
print("std:",std.mean().item())