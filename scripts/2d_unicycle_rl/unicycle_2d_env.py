import math
import torch

class Unicycle2DEnv:
    def __init__(self,num_envs=4096,device="cuda"):
        self.device=torch.device(device)
        self.num_envs=num_envs
        self.dt=0.05
        self.max_v=1.0
        self.max_omega=2.0
        self.max_episode_length=300
        self.map_size=10.0
        self.map_resolution=0.125
        self.map_size_px=80
        self.num_obstacles=1
        self.robot_radius=0.25
        self.curriculum_stage=1
        self.prev_collision=torch.zeros(num_envs,dtype=torch.bool,device=self.device)
        self.stage_thresholds=[0.90,0.85,0.80,0.75]
        self.eval_episodes=200
        self.success_count=0
        self.completed_episodes=0
        self.robot_pos=torch.zeros(num_envs,2,device=self.device)
        self.robot_yaw=torch.zeros(num_envs,device=self.device)
        self.prev_v=torch.zeros(num_envs,device=self.device)
        self.prev_omega=torch.zeros(num_envs,device=self.device)
        self.goal_pos=torch.zeros(num_envs,2,device=self.device)
        self.obstacle_pos=torch.zeros(num_envs,self.num_obstacles,2,device=self.device)
        self.obstacle_size=torch.zeros(num_envs,self.num_obstacles,2,device=self.device)
        self.episode_length=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.prev_goal_dist=torch.zeros(num_envs,device=self.device)
        print(f"[Curriculum] Stage {self.curriculum_stage}")
        self.reset()

    def reset(self,env_ids=None):
        if env_ids is None:
            env_ids=torch.arange(self.num_envs,device=self.device)
        n=len(env_ids)
        self.robot_pos[env_ids]=0.0
        self.prev_v[env_ids]=0.0
        self.prev_omega[env_ids]=0.0
        self.prev_collision[env_ids]=False
        if self.curriculum_stage==0:
            self.robot_yaw[env_ids]=0.0
            self.goal_pos[env_ids]=torch.tensor([4.0,0.0],device=self.device)
            self.obstacle_pos[env_ids]=100.0
            self.obstacle_size[env_ids]=0.0
        elif self.curriculum_stage==1:
            self.robot_yaw[env_ids] = torch.empty(
                n, device=self.device
            ).uniform_(-math.pi / 4, math.pi / 4)
            self.goal_pos[env_ids]=torch.tensor([4.0,0.0],device=self.device)
            self.obstacle_pos[env_ids,0]=torch.tensor([2.0,0.0],device=self.device)
            self.obstacle_size[env_ids,0]=torch.tensor([0.25,0.25],device=self.device)
        elif self.curriculum_stage==2:
            self.robot_yaw[env_ids]=0.0
            self.goal_pos[env_ids]=torch.tensor([4.0,0.0],device=self.device)
            obstacle_pos=torch.empty(n,self.num_obstacles,2,device=self.device).uniform_(-3.0,3.0)
            obstacle_pos[:,:,0]+=1.5
            self.obstacle_pos[env_ids]=obstacle_pos
            self.obstacle_size[env_ids,:,0]=2.0
            self.obstacle_size[env_ids,:,1]=1.0
            self._move_obstacles_away_from_robot(env_ids)
        elif self.curriculum_stage==3:
            self.robot_yaw[env_ids]=0.0
            self.goal_pos[env_ids]=torch.tensor([4.0,0.0],device=self.device)
            obstacle_pos=torch.empty(n,self.num_obstacles,2,device=self.device).uniform_(-3.0,3.0)
            obstacle_pos[:,:,0]+=1.5
            self.obstacle_pos[env_ids]=obstacle_pos
            self.obstacle_size[env_ids,:,0]=torch.empty(n,self.num_obstacles,device=self.device).uniform_(1.0,4.0)
            self.obstacle_size[env_ids,:,1]=torch.empty(n,self.num_obstacles,device=self.device).uniform_(0.5,2.0)
            self._move_obstacles_away_from_robot(env_ids)
        else:
            self.robot_yaw[env_ids]=torch.empty(n,device=self.device).uniform_(-math.pi,math.pi)
            goal=torch.empty(n,2,device=self.device).uniform_(-4.0,4.0)
            goal[:,0]+=2.0
            self.goal_pos[env_ids]=goal
            obstacle_pos=torch.empty(n,self.num_obstacles,2,device=self.device).uniform_(-3.0,3.0)
            obstacle_pos[:,:,0]+=1.5
            self.obstacle_pos[env_ids]=obstacle_pos
            self.obstacle_size[env_ids,:,0]=torch.empty(n,self.num_obstacles,device=self.device).uniform_(1.0,4.0)
            self.obstacle_size[env_ids,:,1]=torch.empty(n,self.num_obstacles,device=self.device).uniform_(0.5,2.0)
            self._move_obstacles_away_from_robot(env_ids)
            self._move_obstacles_away_from_goal(env_ids)
        self.episode_length[env_ids]=0
        self.prev_goal_dist[env_ids]=self.goal_distance()[env_ids]
        return self.get_observations()

    def _move_obstacles_away_from_robot(self,env_ids):
        for _ in range(20):
            d=self.obstacle_pos[env_ids]-self.robot_pos[env_ids,None,:]
            half=self.obstacle_size[env_ids]*0.5+self.robot_radius+0.1
            invalid=(d[...,0].abs()<half[...,0])&(d[...,1].abs()<half[...,1])
            if not invalid.any():
                break
            count=invalid.sum().item()
            new_pos=torch.empty(count,2,device=self.device).uniform_(-3.0,3.0)
            new_pos[:,0]+=1.5
            self.obstacle_pos[env_ids][invalid]=new_pos

    def _move_obstacles_away_from_goal(self,env_ids):
        for _ in range(20):
            d=self.obstacle_pos[env_ids]-self.goal_pos[env_ids,None,:]
            half=self.obstacle_size[env_ids]*0.5+0.1
            invalid=(d[...,0].abs()<half[...,0])&(d[...,1].abs()<half[...,1])
            if not invalid.any():
                break
            count=invalid.sum().item()
            new_pos=torch.empty(count,2,device=self.device).uniform_(-3.0,3.0)
            new_pos[:,0]+=1.5
            self.obstacle_pos[env_ids][invalid]=new_pos

    def step(self,action):
        v=torch.clamp(action[:,0],-1.0,1.0)*self.max_v
        omega=torch.clamp(action[:,1],-1.0,1.0)*self.max_omega
        old_pos=self.robot_pos.clone()
        self.prev_v=v
        self.prev_omega=omega
        self.robot_pos[:,0]+=v*torch.cos(self.robot_yaw)*self.dt
        self.robot_pos[:,1]+=v*torch.sin(self.robot_yaw)*self.dt
        self.robot_yaw+=omega*self.dt
        self.robot_yaw=torch.atan2(torch.sin(self.robot_yaw),torch.cos(self.robot_yaw))
        self.episode_length+=1
        collision=self.check_collision()
        self.robot_pos[collision]=old_pos[collision]
        dist=self.goal_distance()
        progress=self.prev_goal_dist-dist
        self.prev_goal_dist=dist
        goal_reached=dist<0.3
        timeout=self.episode_length>=self.max_episode_length
        collision_forward=collision&(v>0.0)
        collision_stuck=collision&(torch.abs(omega)<0.2)
        collision_escape=self.prev_collision&(~collision)
        reward=progress*5.0
        reward+=goal_reached.float()*20.0
        reward-=collision.float()*20.0
        reward-=collision_forward.float()*5.0
        reward-=collision_stuck.float()*2.0
        reward+=collision_escape.float()*2.0
        reward-=0.01
        done=goal_reached|timeout
        info={"goal":goal_reached,"success":goal_reached,"collision":collision,"stage":self.curriculum_stage}
        self._update_curriculum(done,goal_reached)
        self.prev_collision=collision
        obs=self.get_observations()
        reset_ids=torch.nonzero(done,as_tuple=False).squeeze(-1)
        if reset_ids.numel()>0:
            self.reset(reset_ids)
        return obs,reward,done,info

    def _update_curriculum(self,done,goal_reached):
        done_ids=torch.nonzero(done,as_tuple=False).squeeze(-1)
        if done_ids.numel()==0:
            return
        self.completed_episodes+=done_ids.numel()
        self.success_count+=goal_reached[done_ids].sum().item()
        if self.completed_episodes>=self.eval_episodes:
            success_rate=self.success_count/self.completed_episodes
            if self.curriculum_stage<4:
                threshold=self.stage_thresholds[self.curriculum_stage]
                if success_rate>=threshold:
                    old_stage=self.curriculum_stage
                    # self.curriculum_stage+=1
                    # print(f"[Curriculum] Stage {old_stage} -> {self.curriculum_stage} | success_rate={success_rate:.3f}")
            self.completed_episodes=0
            self.success_count=0

    def goal_distance(self):
        return torch.linalg.vector_norm(self.goal_pos-self.robot_pos,dim=-1)

    def get_local_goal(self):
        d=self.goal_pos-self.robot_pos
        c=torch.cos(self.robot_yaw)
        s=torch.sin(self.robot_yaw)
        local_x=c*d[:,0]+s*d[:,1]
        local_y=-s*d[:,0]+c*d[:,1]
        return torch.stack([local_x,local_y],dim=-1)

    def check_collision(self):
        d=self.robot_pos[:,None,:]-self.obstacle_pos
        half=self.obstacle_size*0.5+self.robot_radius
        inside=(d[...,0].abs()<half[...,0])&(d[...,1].abs()<half[...,1])
        return inside.any(dim=1)

    def get_occupancy_map(self):
        N=self.num_envs
        W=self.map_size_px
        occupancy=torch.zeros(N,W,W,dtype=torch.float32,device=self.device)
        d=self.obstacle_pos-self.robot_pos[:,None,:]
        c=torch.cos(self.robot_yaw)[:,None]
        s=torch.sin(self.robot_yaw)[:,None]
        local_x=c*d[...,0]+s*d[...,1]
        local_y=-s*d[...,0]+c*d[...,1]
        center=W/2.0
        cx=local_x/self.map_resolution+center
        cy=local_y/self.map_resolution+center
        grid=torch.arange(W,device=self.device)
        gx=grid[None,None,:]
        gy=grid[None,:,None]
        for k in range(self.num_obstacles):
            hx=self.obstacle_size[:,k,0]/self.map_resolution/2.0
            hy=self.obstacle_size[:,k,1]/self.map_resolution/2.0
            mask=(gx>=cx[:,k,None,None]-hx[:,None,None])&(gx<=cx[:,k,None,None]+hx[:,None,None])&(gy>=cy[:,k,None,None]-hy[:,None,None])&(gy<=cy[:,k,None,None]+hy[:,None,None])
            occupancy=torch.maximum(occupancy,mask.float())
        return occupancy

    def get_observations(self):
        occupancy=self.get_occupancy_map()
        ray_heightmap=occupancy.unsqueeze(1)
        goal=self.get_local_goal()
        local_lin_vel=torch.stack([self.prev_v,torch.zeros_like(self.prev_v),torch.zeros_like(self.prev_v)],dim=-1)
        local_ang_vel=torch.stack([torch.zeros_like(self.prev_omega),torch.zeros_like(self.prev_omega),self.prev_omega],dim=-1)
        height=torch.zeros(self.num_envs,1,device=self.device)
        heading_sin=torch.sin(self.robot_yaw).unsqueeze(-1)
        heading_cos=torch.cos(self.robot_yaw).unsqueeze(-1)
        policy_obs=torch.cat([local_lin_vel,local_ang_vel,height,heading_sin,heading_cos,goal],dim=-1)
        assert ray_heightmap.shape==(self.num_envs,1,80,80)
        assert policy_obs.shape==(self.num_envs,11)
        return {"ray_heightmap":ray_heightmap,"policy_obs":policy_obs}