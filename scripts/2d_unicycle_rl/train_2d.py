import os
import argparse
from datetime import datetime
import torch
import gymnasium as gym
from skrl.envs.wrappers.torch import Wrapper
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO,PPO_CFG
from skrl.trainers.torch import SequentialTrainer
from unicycle_2d_env import Unicycle2DEnv
from models import Policy,Value

NUM_ENVS=1024
DEVICE="cuda"
TIMESTEPS=2000
ROLLOUTS=8
LEARNING_EPOCHS=3
MINI_BATCHES=16
LEARNING_RATE=3.0e-4

class Unicycle2DTrainEnv(Wrapper):
    def __init__(self,num_envs=1024,device="cuda"):
        self._env=Unicycle2DEnv(num_envs=num_envs,device=device)
        self._num_envs=num_envs
        self._num_agents=1
        self._device=torch.device(device)
        self._observation_space=gym.spaces.Box(low=-float("inf"),high=float("inf"),shape=(6411,),dtype=float)
        self._action_space=gym.spaces.Box(low=-1.0,high=1.0,shape=(2,),dtype=float)
        self._state_space=None
        self.step_count=0
    @property
    def num_envs(self):
        return self._num_envs
    @property
    def num_agents(self):
        return self._num_agents
    @property
    def device(self):
        return self._device
    @property
    def observation_space(self):
        return self._observation_space
    @property
    def action_space(self):
        return self._action_space
    @property
    def state_space(self):
        return self._state_space
    def _flatten_obs(self,obs):
        heightmap=obs["ray_heightmap"].reshape(self._num_envs,-1)
        policy_obs=obs["policy_obs"]
        return torch.cat([heightmap,policy_obs],dim=-1)
    def reset(self):
        obs=self._env.reset()
        return self._flatten_obs(obs),{}
    def step(self,actions):
        obs,reward,done,info=self._env.step(actions)
        reward=reward.reshape(self._num_envs,1)
        terminated=done.reshape(self._num_envs,1)
        truncated=torch.zeros_like(terminated)
        self.step_count+=1
        if self.step_count%100==0:
            success_rate=info["success"].float().mean().item()
            collision_rate=info["collision"].float().mean().item()
            print(f"[Train] step={self.step_count} reward_mean={reward.mean().item():.4f} reward_max={reward.max().item():.4f} reward_min={reward.min().item():.4f} success={success_rate:.3f} collision={collision_rate:.3f}")
        return self._flatten_obs(obs),reward,terminated,truncated,info
    def close(self):
        pass
    def render(self,*args,**kwargs):
        return None
    def state(self):
        return None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=str,default=None)
    args=parser.parse_args()
    timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir=os.path.join("logs/skrl/unicycle_navigation_2d",timestamp)
    os.makedirs(output_dir,exist_ok=True)
    print(f"Output directory: {output_dir}")
    env=Unicycle2DTrainEnv(num_envs=NUM_ENVS,device=DEVICE)
    print(f"num_envs: {env.num_envs}")
    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")
    obs,_=env.reset()
    print(f"observations: {obs.shape}")
    memory=RandomMemory(memory_size=ROLLOUTS,num_envs=env.num_envs,device=env.device)
    models={}
    models["policy"]=Policy(env.observation_space,env.state_space,env.action_space,env.device).to(env.device)
    models["value"]=Value(env.observation_space,env.state_space,env.action_space,env.device).to(env.device)
    cfg_agent=PPO_CFG()
    cfg_agent.rollouts=ROLLOUTS
    cfg_agent.learning_epochs=LEARNING_EPOCHS
    cfg_agent.mini_batches=MINI_BATCHES
    cfg_agent.discount_factor=0.99
    cfg_agent.gae_lambda=0.95
    cfg_agent.learning_rate=LEARNING_RATE
    cfg_agent.grad_norm_clip=1.0
    cfg_agent.ratio_clip=0.2
    cfg_agent.value_clip=0.2
    cfg_agent.entropy_loss_scale=0.005
    cfg_agent.value_loss_scale=1.0
    cfg_agent.time_limit_bootstrap=False
    cfg_agent.experiment.directory=output_dir
    cfg_agent.experiment.experiment_name=""
    cfg_agent.experiment.write_interval=100
    cfg_agent.experiment.checkpoint_interval=1000
    agent=PPO(models=models,memory=memory,cfg=cfg_agent,observation_space=env.observation_space,state_space=env.state_space,action_space=env.action_space,device=env.device)
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
        print("Checkpoint loaded")
        print("Policy, value and optimizer restored")
    else:
        print("Training from scratch")
    trainer_cfg={"timesteps":TIMESTEPS,"headless":True,"environment_info":"log"}
    trainer=SequentialTrainer(env=env,agents=agent,cfg=trainer_cfg)
    trainer.train()
    env.close()

if __name__=="__main__":
    main()