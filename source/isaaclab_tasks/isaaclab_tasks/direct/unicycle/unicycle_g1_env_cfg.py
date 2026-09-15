from isaaclab.utils import configclass
from .unicycle_env_cfg import UnicycleEnvCfg
from agile.rl_env.assets.robots import unitree_g1
from isaaclab.assets import ArticulationCfg

@configclass
class UnicycleG1EnvCfg(UnicycleEnvCfg):
    g1 = unitree_g1.G1_29DOF_DELAYED_DC_MOTOR.replace(
        prim_path="/World/envs/env_.*/G1"
    )
    g1.init_state.pos = (-2.0, 0.0, 0.9)