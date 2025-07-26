# aloha_scripts/constants.py

TASK_CONFIGS = {
    "bin-picking-v0": {
        "dataset_dir": "./data/bin-picking-v0",
        "num_episodes": 1000,
        "episode_len": 100,
        "camera_names": ["agentview", "eye_in_hand"],
        "camera_resolution": [128, 128]
    },
    "peg-insertion-v0": {
        "dataset_dir": "./data/peg-insertion-v0",
        "num_episodes": 1000,
        "episode_len": 100,
        "camera_names": ["agentview"],
        "camera_resolution": [128, 128]
    },
    "sim_transfer_cube_scripted": {
        "dataset_dir": "./data/sim_transfer_cube_scripted",
        "num_episodes": 50,
        "episode_len": 200,
        "camera_names": ["main"],
        "camera_resolution": [128, 128]
    }
}
