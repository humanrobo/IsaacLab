import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

BRUCE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",  # ← これが絶対必要！
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/matsuno/IsaacLab/robots/bruce.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=8,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0,0.0,0.1),
        joint_pos={
                    "hip_yaw_l": 0.0,
                    "hip_yaw_r": 0.0,
                    "hip_pitch_l": 0.0,   # 【変更】いったんまっすぐ（0.0）にするか、自然な姿勢にする
                    "hip_pitch_r": 0.0,
                    "knee_pitch_l": 0.0,  # 【変更】
                    "knee_pitch_r": 0.0,  # 【変更】
                    "ankle_pitch_l": 0.0, # 【変更】
                    "ankle_pitch_r": 0.0, # 【変更】
                    "shoulder_pitch_l": 0.0,
                    "shoulder_pitch_r": 0.0,
                    "shoulder_roll_l": 0.0,
                    "shoulder_roll_r": 0.0,
                    "hip_roll_l": 0.0,
                    "hip_roll_r": 0.0,
                    "elbow_pitch_l": 0.0,
                    "elbow_pitch_r": 0.0,
                },
                joint_vel={".*": 0.0},
            ),

    soft_joint_pos_limit_factor=0.9,

    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=200.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)