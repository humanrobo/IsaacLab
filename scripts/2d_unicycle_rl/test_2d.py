import os
import time
from datetime import datetime
from unicycle_2d_env import Unicycle2DEnv
from skrl.utils.runner.torch import Runner

NUM_ENVS=4096
DEVICE="cuda"
AGENT_CFG="config.yaml"

def main():
    start_time=time.time()
    env=Unicycle2DEnv(num_envs=NUM_ENVS,device=DEVICE)
    print(f"[INFO] num_envs: {NUM_ENVS}")
    print(f"[INFO] device: {DEVICE}")
    obs=env.reset()
    print(f"[INFO] ray_heightmap: {obs['ray_heightmap'].shape}")
    print(f"[INFO] policy_obs: {obs['policy_obs'].shape}")
    print("[INFO] 2D environment initialized.")
    # TODO: skrl environment wrapper
    # TODO: skrl Runner
    # runner=Runner(env,agent_cfg)
    # runner.run()
    print(f"Initialization time: {round(time.time()-start_time,2)} seconds")

if __name__=="__main__":
    main()