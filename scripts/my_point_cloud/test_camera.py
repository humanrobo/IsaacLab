from isaaclab.app import AppLauncher


# -------------------------------------------------------
# Launch Isaac Sim
# -------------------------------------------------------
app_launcher = AppLauncher(
    headless=False,
    enable_cameras=True,
)
simulation_app = app_launcher.app

# -------------------------------------------------------
# Imports
# -------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from pxr import UsdGeom

# -------------------------------------------------------
# Simulation
# -------------------------------------------------------
sim_cfg = sim_utils.SimulationCfg(dt=0.01)
sim = sim_utils.SimulationContext(sim_cfg)

# Ground
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/GroundPlane", cfg)

# -------------------------------------------------------
# Camera parent Xform
# -------------------------------------------------------
UsdGeom.Xform.Define(
    sim.stage,
    "/World/CameraRig"
)

# -------------------------------------------------------
# Camera
# -------------------------------------------------------
camera_cfg = CameraCfg(
    prim_path="/World/CameraRig/MyCamera",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb"],

    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        clipping_range=(0.1,100.0),
    ),

    offset=CameraCfg.OffsetCfg(
        pos=(2.0,0.0,1.5),
        rot=(1.0,0.0,0.0,0.0),
        convention="ros",
    ),
)

camera = Camera(camera_cfg)
from pxr import Usd

print("===== Stage =====")
for prim in sim.stage.Traverse():
    print(prim.GetPath())
print("=================")
# -------------------------------------------------------
# Start simulation
# -------------------------------------------------------
sim.reset()
from pxr import UsdGeom
prim = sim.stage.GetPrimAtPath("/World/CameraRig/MyCamera")

print("Valid :", prim.IsValid())

if prim.IsValid():
    print("Type  :", prim.GetTypeName())
camera_prim = UsdGeom.Camera(
    sim.stage.GetPrimAtPath("/World/CameraRig/MyCamera")
)

transform = camera_prim.ComputeLocalToWorldTransform(0.0)
print(transform)

while simulation_app.is_running():
    sim.step()
    sim.render()

    # カメラデータ更新
    camera.update(sim.get_physics_dt())
    print(camera.is_initialized)
    print("pos_w      :", camera.data.pos_w[0].cpu().numpy())
    print("quat_world :", camera.data.quat_w_world[0].cpu().numpy())
    print("quat_ros   :", camera.data.quat_w_ros[0].cpu().numpy())
    print(camera.data)

    # 世界座標
    pos = camera.data.pos_w[0]
    quat = camera.data.quat_w_ros[0]   # (x,y,z,w)

    print("=" * 60)
    print("Position :", pos.cpu().numpy())
    print("Quaternion (ROS x,y,z,w):", quat.cpu().numpy())