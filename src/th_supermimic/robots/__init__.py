"""Per-process robot selection, before constructing any tensors or task configs."""
from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joint_names: tuple[str, ...]
    tracked_bodies: tuple[str, ...]
    root_body: str
    context_body: str
    root_height: float
    default_joint_pos: tuple[float, ...]
    motors: tuple[dict, ...]

    @property
    def num_joints(self):
        return len(self.joint_names)

    @property
    def context_dim(self):
        return self.num_joints + 4


def get_robot(name):
    if name not in ('k1', 'g1'):
        raise ValueError(f'Unknown robot {name!r}; choose k1 or g1')
    raw = json.loads((Path(__file__).parent / f'{name}.json').read_text())
    return RobotSpec(raw['robot'], tuple(raw['joint_names']), tuple(raw['tracked_bodies']),
                     raw['root_body'], raw['context_body'], raw['root_height'],
                     tuple(raw['default_joint_pos']), tuple(raw['motors']))


ROBOT = get_robot(os.environ.get('TH_SUPERMIMIC_ROBOT', 'k1'))
