from pathlib import Path
import numpy as np
import pytest
from th_supermimic.data.rmai import convert_file, resample
from th_supermimic.robots import get_robot


@pytest.mark.parametrize('robot', ['k1', 'g1'])
def test_reorders_joints_and_preserves_quaternion_rotation(tmp_path, robot):
    s = get_robot(robot)
    q = np.zeros((31, 7+s.num_joints))
    q[:, 2] = s.root_height
    q[:, 3] = 1
    q[1::2, 3] = -1  # equivalent rotation; avoid long-path interpolation
    q[:, 7:] = np.arange(s.num_joints)
    names = list(reversed(s.joint_names))
    p = tmp_path/'robot_motion.npz'
    np.savez(p, qpos=q, joint_names=names, fps=30, robot_id=robot)
    csv, metadata = convert_file(p, robot)
    assert csv.shape == (51, 7+s.num_joints)
    assert metadata['fps'] == 50
    np.testing.assert_allclose(csv[:, 7:], np.tile(np.arange(s.num_joints)[::-1], (51,1)))
    np.testing.assert_allclose(np.abs(csv[:, 6]), 1)
    np.testing.assert_allclose(csv[:, 3:6], 0)
    with pytest.raises(ValueError, match='wrong qpos width'):
        convert_file(p, 'k1' if robot=='g1' else 'g1')


def test_missing_names_and_invalid_quaternion_are_rejected(tmp_path):
    p = tmp_path/'bad.npz'
    q = np.zeros((4,30));q[:,3] = 1
    np.savez(p, qpos=q, fps=30)
    with pytest.raises(ValueError, match='joint_names required'):
        convert_file(p,'k1')
    q[:,3] = 0
    with pytest.raises(ValueError, match='zero-norm'):
        resample(q,30)
