RLDS_TO_LEROBOT_DATASET_CONFIGS = {
    "toto": {
        "robot": "panda",
        "image_size": (480, 640),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (480, 640, 3),
            },
            {
                "rlds_path": "/steps/observation/natural_language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (7,),
            },
        ],
    },

    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "robot": "panda",
        "image_size": (128, 128),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (128, 128, 3),
            },
            {
                "rlds_path": "/steps/language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (13,),
            },
        ],
    },

    "utaustin_mutex": {
        "robot": "panda",
        "image_size": (128, 128),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (128, 128, 3),
            },
            {
                "rlds_path": "/steps/language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (24,),
            },
        ],
    },

    "berkeley_autolab_ur5": {
        "robot": "ur5e",
        "image_size": (480, 640),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (480, 640, 3),
            },
            {
                "rlds_path": "/steps/observation/natural_language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/robot_state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (15,),
            },
        ],
    },

    "kaist_nonprehensile_converted_externally_to_rlds": {
        "robot": "panda",
        "image_size": (480, 640),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (480, 640, 3),
            },
            {
                "rlds_path": "/steps/language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (21,),
            },
        ],
    },

    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "robot": "panda",
        "image_size": (360, 640),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (360, 640, 3),
            },
            {
                "rlds_path": "/steps/language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (20,),
            },
        ],
    },

    "fractal20220817_data": {
        "robot": "google_robot",
        "image_size": (256, 320),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (256, 320, 3),
            },
            {
                "rlds_path": "/steps/observation/natural_language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/base_pose_tool_reached",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (7,),
            },
        ],
    },

    "viola": {
        "robot": "panda",
        "image_size": (224, 224),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/agentview_rgb",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (224, 224, 3),
            },
            {
                "rlds_path": "/steps/observation/natural_language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/ee_states",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (16,),
            },
        ],
    },

    "taco_play": {
        "robot": "panda",
        "image_size": (150, 200),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/rgb_static",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (150, 200, 3),
            },
            {
                "rlds_path": "/steps/observation/natural_language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/robot_obs",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (15,),
            },
        ],
    },

    "bridge": {
        "robot": "widowX",
        "image_size": (480, 640),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (480, 640, 3),
            },
            {
                "rlds_path": "/steps/observation/natural_language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (7,),
            },
        ],
    },

    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "robot": "xarm7",
        "image_size": (480, 640),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (480, 640, 3),
            },
            {
                "rlds_path": "/steps/language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (21,),
            },
        ],
    },

    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "robot": "xarm7",
        "image_size": (224, 224),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (224, 224, 3),
            },
            {
                "rlds_path": "/steps/language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/end_effector_pose",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (6,),
            },
        ],
    },

    "jaco_play": {
        "robot": "jaco",
        "image_size": (224, 224),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (224, 224, 3),
            },
            {
                "rlds_path": "/steps/observation/natural_language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/end_effector_cartesian_pos",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (7,),
            },
        ],
    },

    "austin_sailor_dataset_converted_externally_to_rlds": {
        "robot": "panda",
        "image_size": (128, 128),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (128, 128, 3),
            },
            {
                "rlds_path": "/steps/language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (8,),
            },
        ],
    },

    "language_table": {
        "robot": "xarm7",
        "image_size": (360, 640),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/rgb",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (360, 640, 3),
            },
            {
                "rlds_path": "/steps/observation/instruction",
                "lerobot_path": "language_instruction_tokens",
                "dtype": "int32",
                "shape": (512,),
            },
            {
                "rlds_path": "/steps/observation/effector_translation",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (2,),
            },
        ],
    },

    "austin_buds_dataset_converted_externally_to_rlds": {
        "robot": "panda",
        "image_size": (128, 128),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (128, 128, 3),
            },
            {
                "rlds_path": "/steps/language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (24,),
            },
        ],
    },
}

ROBOT_JOINT_NUMBERS = {
    "panda": 7,
    "ur5e": 6,
    "widowX": 6,
    "xarm7": 7,
    "jaco": 6,
    "kuka_iiwa": 7,
    "kinova3": 7,
    "sawyer": 7,
    "google_robot": 7
}