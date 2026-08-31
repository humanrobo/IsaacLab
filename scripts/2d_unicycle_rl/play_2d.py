import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import argparse

CHECKPOINT="/home/matsuno/IsaacLab/logs/unicycle_navigation_2d/checkpoints/26-08-31_16-40-29-033861_PPO/checkpoints/best_agent.pt"
DEVICE="cuda"
DT=0.05
MAX_STEPS=1000

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
            nn.Flatten()
        )
        self.net=nn.Sequential(
            nn.Linear(2059,256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU()
        )
        self.mean_layer=nn.Linear(128,2)
        self.log_std_parameter=nn.Parameter(torch.full((2,),-1.0))

    def forward(self,obs):
        heightmap=obs[:,:6400].reshape(-1,1,80,80)
        policy_obs=obs[:,6400:6411]
        x=self.heightmap_features(heightmap)
        x=torch.cat([x,policy_obs],dim=-1)
        x=self.net(x)
        return self.mean_layer(x)

class Unicycle2D:
    def __init__(self,device="cuda"):
        self.device=torch.device(device)
        self.dt=DT
        self.max_v=1.0
        self.max_omega=2.0
        self.map_resolution=0.125
        self.map_size_px=80
        self.robot_radius=0.25
        self.reset()

    def reset(self):
        self.robot_pos=torch.zeros(2,device=self.device)
        self.robot_yaw=torch.empty((),device=self.device).uniform_(-math.pi,math.pi)
        self.prev_v=torch.tensor(0.0,device=self.device)
        self.prev_omega=torch.tensor(0.0,device=self.device)
        self.goal_pos=torch.empty(2,device=self.device).uniform_(-4.0,4.0)
        self.goal_pos[0]+=2.0
        self.obstacle_pos=torch.empty(2,device=self.device).uniform_(-3.5,3.5)
        self.obstacle_pos[0]+=1.5
        self.obstacle_size=torch.tensor([
            torch.empty((),device=self.device).uniform_(1.0,7.0),
            torch.empty((),device=self.device).uniform_(0.5,2.0)
        ],device=self.device)
        self.prev_goal_dist=self.goal_distance()
        print("robot:",self.robot_pos.cpu().numpy())
        print("goal:",self.goal_pos.cpu().numpy())
        print("obstacle:",self.obstacle_pos.cpu().numpy())
        print("obstacle_size:",self.obstacle_size.cpu().numpy())
        print("collision:",self.check_collision())
        return self.get_observations()

    def goal_distance(self):
        return torch.linalg.vector_norm(self.goal_pos-self.robot_pos)

    def get_local_goal(self):
        d=self.goal_pos-self.robot_pos
        c=torch.cos(self.robot_yaw)
        s=torch.sin(self.robot_yaw)
        local_x=c*d[0]+s*d[1]
        local_y=-s*d[0]+c*d[1]
        return torch.stack([local_x,local_y])

    def check_collision(self):
        d=self.robot_pos-self.obstacle_pos
        half=self.obstacle_size*0.5+self.robot_radius
        return bool((d[0].abs()<half[0])&(d[1].abs()<half[1]))

    def get_occupancy_map(self):
        W=self.map_size_px
        occupancy=torch.zeros(W,W,device=self.device)
        d=self.obstacle_pos-self.robot_pos
        c=torch.cos(self.robot_yaw)
        s=torch.sin(self.robot_yaw)
        local_x=c*d[0]+s*d[1]
        local_y=-s*d[0]+c*d[1]
        center=W/2.0
        cx=local_x/self.map_resolution+center
        cy=local_y/self.map_resolution+center
        hx=self.obstacle_size[0]/self.map_resolution/2.0
        hy=self.obstacle_size[1]/self.map_resolution/2.0
        grid=torch.arange(W,device=self.device)
        gx=grid[None,:]
        gy=grid[:,None]
        mask=(gx>=cx-hx)&(gx<=cx+hx)&(gy>=cy-hy)&(gy<=cy+hy)
        occupancy[mask]=1.0
        return occupancy

    def get_observations(self):
        occupancy=self.get_occupancy_map()
        goal=self.get_local_goal()
        local_lin_vel=torch.stack([
            self.prev_v,
            torch.tensor(0.0,device=self.device),
            torch.tensor(0.0,device=self.device)
        ])
        local_ang_vel=torch.stack([
            torch.tensor(0.0,device=self.device),
            torch.tensor(0.0,device=self.device),
            self.prev_omega
        ])
        height=torch.tensor([0.0],device=self.device)
        heading_sin=torch.sin(self.robot_yaw).reshape(1)
        heading_cos=torch.cos(self.robot_yaw).reshape(1)
        policy_obs=torch.cat([
            local_lin_vel,
            local_ang_vel,
            height,
            heading_sin,
            heading_cos,
            goal
        ])
        obs=torch.cat([occupancy.reshape(-1),policy_obs])
        return obs.unsqueeze(0)

    def step(self,action):
        v=torch.clamp(action[0],-1.0,1.0)*self.max_v
        omega=torch.clamp(action[1],-1.0,1.0)*self.max_omega
        old_pos=self.robot_pos.clone()
        self.prev_v=v
        self.prev_omega=omega
        self.robot_pos[0]+=v*torch.cos(self.robot_yaw)*self.dt
        self.robot_pos[1]+=v*torch.sin(self.robot_yaw)*self.dt
        self.robot_yaw+=omega*self.dt
        self.robot_yaw=torch.atan2(torch.sin(self.robot_yaw),torch.cos(self.robot_yaw))
        collision=self.check_collision()
        self.robot_pos[collision]=old_pos[collision]
        dist=self.goal_distance()
        reward=(self.prev_goal_dist-dist)*5.0
        self.prev_goal_dist=dist
        goal=dist<0.3
        reward+=float(goal)*20.0
        reward-=float(collision)*20.0
        reward-=0.01
        done=goal
        return self.get_observations(),reward,done

def draw(ax,env,trajectory,step):
    ax.clear()
    ax.set_xlim(-5,7)
    ax.set_ylim(-5,5)
    ax.set_aspect("equal")
    ax.grid(True)
    obstacle_x=env.obstacle_pos[0].item()
    obstacle_y=env.obstacle_pos[1].item()
    obstacle_w=env.obstacle_size[0].item()
    obstacle_h=env.obstacle_size[1].item()
    rect=plt.Rectangle(
        (obstacle_x-obstacle_w/2,obstacle_y-obstacle_h/2),
        obstacle_w,
        obstacle_h,
        fill=True,
        alpha=0.5
    )
    ax.add_patch(rect)
    robot_x=env.robot_pos[0].item()
    robot_y=env.robot_pos[1].item()
    ax.plot(robot_x,robot_y,"bo",markersize=10)
    yaw=env.robot_yaw.item()
    ax.arrow(
        robot_x,
        robot_y,
        0.5*math.cos(yaw),
        0.5*math.sin(yaw),
        head_width=0.12,
        length_includes_head=True
    )
    goal_x=env.goal_pos[0].item()
    goal_y=env.goal_pos[1].item()
    ax.plot(goal_x,goal_y,"r*",markersize=15)
    if trajectory:
        traj=np.asarray(trajectory)
        ax.plot(traj[:,0],traj[:,1],"b-",alpha=0.7)
    ax.set_title(f"step={step}  dist={env.goal_distance().item():.2f}")
    plt.pause(0.001)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=str,required=True)
    args=parser.parse_args()
    print("Loading policy...")
    policy=Policy().to(DEVICE)
    checkpoint=torch.load(args.checkpoint,map_location=DEVICE)
    policy.load_state_dict(checkpoint["policy"],strict=True)
    policy.eval()
    print(f"Loaded: {args.checkpoint}")
    env=Unicycle2D(DEVICE)
    env.reset()
    env.robot_pos[:]=torch.tensor([0.0,0.0],device=env.device)
    env.robot_yaw.fill_(0.0)
    env.goal_pos[:]=torch.tensor([3.0,3.0],device=env.device)
    env.obstacle_pos[:]=torch.tensor([6.0,0.0],device=env.device)
    env.obstacle_size[:]=torch.tensor([1.0,2.0],device=env.device)
    obs=env.get_observations()
    trajectory=[
        env.robot_pos.detach().cpu().numpy().copy()
    ]
    plt.ion()
    fig,ax=plt.subplots(figsize=(8,8))
    for step in range(MAX_STEPS):
        with torch.no_grad():
            mean=policy(obs)
            action=torch.clamp(mean,-1.0,1.0)[0]
        obs,reward,done=env.step(action)
        trajectory.append(
            env.robot_pos.detach().cpu().numpy().copy()
        )
        draw(ax,env,trajectory,step)
        if step%10==0:
            print(
                f"step={step} "
                f"action={action.cpu().numpy()} "
                f"reward={reward:.4f} "
                f"dist={env.goal_distance().item():.3f}"
            )
        if done:
            print("episode done")
            break
    plt.ioff()
    plt.show()

if __name__=="__main__":
    main()