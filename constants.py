import os

DT = 0.05
PUPPET_GRIPPER_JOINT_OPEN = [0.04, 0.04]
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: 0.04 * (x + 1.0) / 2.0
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: 2.0 * x / 0.04 - 1.0
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / 10.0
MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: 2.0 * x / 0.04 - 1.0
XML_DIR = os.path.join(os.path.dirname(__file__), "assets", "xml")
START_ARM_POSE = [0.0, -0.5, 0.0, 1.5, 0.0, 1.0, 0.0]

SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted': {
        'dataset_dir': './data/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 200,
        'camera_names': ['agentview', 'eye_in_hand'],
        'name_filter': lambda name: True,
    }
}
