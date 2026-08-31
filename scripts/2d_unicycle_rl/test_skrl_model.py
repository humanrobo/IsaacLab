import torch
from gymnasium import spaces
from models import Policy,Value

device="cuda"
observation_space=spaces.Box(low=-float("inf"),high=float("inf"),shape=(6411,),dtype=float)
state_space=spaces.Box(low=-float("inf"),high=float("inf"),shape=(6411,),dtype=float)
action_space=spaces.Box(low=-1.0,high=1.0,shape=(2,),dtype=float)

policy=Policy(observation_space,state_space,action_space,device).to(device)
value=Value(observation_space,state_space,action_space,device).to(device)

observations=torch.zeros(4096,6411,device=device)
inputs={"observations":observations}

with torch.no_grad():
    actions,policy_info=policy.act(inputs,role="policy")
    values,_=value.act(inputs,role="value")

print("actions:",actions.shape)
print("values:",values.shape)
print("log_std:",policy_info["log_std"].shape)