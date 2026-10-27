"""Active robot adapter; select TH_SUPERMIMIC_ROBOT before importing the task."""
import json
import os
from pathlib import Path
from th_supermimic.robots import ROBOT

JOINT_NAMES = list(ROBOT.joint_names)
TRACKED_BODIES = list(ROBOT.tracked_bodies)
# Prepared metadata accompanies the generated MJCF. No fallback to G1 geoms.
metadata = Path(os.environ.get('TH_SUPERMIMIC_ASSETS', f'assets/{ROBOT.name}')) / 'metadata.json'
COLLISION_GEOM_NAMES = tuple(json.loads(metadata.read_text())['collision_geom_names']) if metadata.is_file() else ()


def action_scale(joint_name):
    return ROBOT.motors[JOINT_NAMES.index(joint_name)]['action_scale']


def default_joint_pos_list():
    return list(ROBOT.default_joint_pos)
