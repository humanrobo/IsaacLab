
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

from isaaclab_assets.robots.bruce import BRUCE_CFG


def main():

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device="cuda:0"
    )

    sim = SimulationContext(sim_cfg)

    robot = Articulation(
        BRUCE_CFG.replace(
            prim_path="/World/Bruce"
        )
    )

    sim.reset()

    print(robot.data.joint_names)

    while True:
        sim.step()


if __name__ == "__main__":
    main()