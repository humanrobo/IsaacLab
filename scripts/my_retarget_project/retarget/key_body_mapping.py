#humanoid_walk.npzとh1.urdfの対応関係を確認するスクリプト
# key_body_mapping.py

# key_body_mapping.py

# retarget/key_body_mapping.py

HUMANOID_TO_H1 = {

    # torso
    "torso": "torso_link",

    # legs
    "right_foot": "right_ankle_link",
    "left_foot": "left_ankle_link",

    "right_shin": "right_knee_link",
    "left_shin": "left_knee_link",

    # arms (elbowまで)
    "right_lower_arm": "right_elbow_link",
    "left_lower_arm": "left_elbow_link",
}


H1_JOINTS = [

"left_hip_yaw_joint",
"left_hip_roll_joint",
"left_hip_pitch_joint",
"left_knee_joint",
"left_ankle_joint",

"right_hip_yaw_joint",
"right_hip_roll_joint",
"right_hip_pitch_joint",
"right_knee_joint",
"right_ankle_joint",

"torso_joint",

"left_shoulder_pitch_joint",
"left_shoulder_roll_joint",
"left_shoulder_yaw_joint",
"left_elbow_joint",

"right_shoulder_pitch_joint",
"right_shoulder_roll_joint",
"right_shoulder_yaw_joint",
"right_elbow_joint",

]

TARGET_BODIES = [

    # torso
    "torso",

    # legs
    "right_foot",
    "left_foot",

    "right_shin",
    "left_shin",

    # arms
    "right_lower_arm",
    "left_lower_arm",
]