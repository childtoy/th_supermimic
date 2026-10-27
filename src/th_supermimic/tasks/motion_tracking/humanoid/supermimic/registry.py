"""Register the G1 SuperMimic task with UniLab."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from th_supermimic.tasks.motion_tracking.humanoid.supermimic.env import (
    SUPPORTED_BACKENDS,
    make_supermimic_env,
)

TASK_NAME = "THSuperMimic"

registry.register_env_config(TASK_NAME, ManagerBasedRlEnvCfg)
for backend in SUPPORTED_BACKENDS:
    registry.register_env(TASK_NAME, make_supermimic_env, sim_backend=backend)

__all__ = ["TASK_NAME"]
