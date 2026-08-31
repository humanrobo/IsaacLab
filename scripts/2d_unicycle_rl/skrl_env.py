import torch
from unicycle_2d_env import Unicycle2DEnv

class Unicycle2DSkrlEnv:
    def __init__(self,num_envs=4096,device="cuda"):
        self.env=Unicycle2DEnv(num_envs=num_envs,device=device)
        self.num_envs=num_envs
        self.num_agents=1
        self.device=torch.device(device)
        self.observation_space={
            "ray_heightmap":(1,80,80),
            "policy_obs":(11,),
        }
        self.action_space=(2,)
    def reset(self):
        return self.env.reset()
    def step(self,actions):
        obs,reward,done,info=self.env.step(actions)
        return obs,reward,done,info
    def close(self):
        pass
    def render(self):
        return None
    def state(self):
        return None

if __name__=="__main__":
    env=Unicycle2DSkrlEnv(num_envs=4096,device="cuda")
    obs=env.reset()
    print("num_envs:",env.num_envs)
    print("device:",env.device)
    print("ray_heightmap:",obs["ray_heightmap"].shape)
    print("policy_obs:",obs["policy_obs"].shape)
    actions=torch.zeros(4096,2,device="cuda")
    obs,reward,done,info=env.step(actions)
    print("after step:")
    print("ray_heightmap:",obs["ray_heightmap"].shape)
    print("policy_obs:",obs["policy_obs"].shape)
    print("reward:",reward.shape)
    print("done:",done.shape)