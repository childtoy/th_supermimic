"""RIMKit robot qpos (Z-up, wxyz) to named, 50 Hz SuperMimic CSV clips."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from th_supermimic.robots import get_robot


def resample(qpos, source_fps, target_fps=50.0):
    if qpos.ndim != 2 or len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError('qpos must contain at least two finite frames')
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError('invalid source fps')
    quat = qpos[:, 3:7].copy()
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('zero-norm root quaternion')
    quat /= norm
    count = math.floor((len(qpos) - 1) * target_fps / source_fps + 1e-8) + 1
    t = np.arange(count) * source_fps / target_fps
    lo = np.minimum(np.floor(t).astype(int), len(qpos) - 2)
    alpha = (t - lo)[:, None]
    out = (1 - alpha) * qpos[lo] + alpha * qpos[lo + 1]
    a, b = quat[lo], quat[lo + 1]
    dot = np.sum(a * b, axis=1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.abs(dot).clip(0, 1)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    near = dot > .9995
    wa = np.where(near, 1 - alpha, np.sin((1-alpha)*theta) / np.maximum(sin, 1e-10))
    wb = np.where(near, alpha, np.sin(alpha*theta) / np.maximum(sin, 1e-10))
    out[:, 3:7] = wa * a + wb * b
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def convert_file(path, robot):
    spec = get_robot(robot)
    with np.load(path, allow_pickle=False) as z:
        qpos = np.asarray(z['qpos'], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 7 + spec.num_joints:
            raise ValueError(f'{path}: wrong qpos width for {robot}: {qpos.shape}')
        if 'joint_names' not in z:
            raise ValueError(f'{path}: joint_names required; refusing to guess qpos ordering')
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in z['joint_names']]
        if len(names) != len(set(names)) or set(names) != set(spec.joint_names):
            raise ValueError(f'{path}: joint_names do not match {robot}')
        if 'robot_id' in z and str(z['robot_id'].item()) != robot:
            raise ValueError(f'{path}: robot_id mismatch')
        if 'qpos_layout' in z and str(z['qpos_layout'].item()) != 'root_xyz_quat_wxyz_then_model_joint_qpos':
            raise ValueError(f'{path}: unsupported qpos_layout')
        fps = float(np.asarray(z['fps']).reshape(-1)[0])
        time_key = 'timestamps_s' if 'timestamps_s' in z else 'timestamps'
        if time_key in z:
            times = np.asarray(z[time_key])
            if times.shape != (len(qpos),) or not np.allclose(np.diff(times), 1 / fps, atol=1e-5):
                raise ValueError(f'{path}: timestamps disagree with fps')
    qpos = qpos[:, [*range(7), *[7 + names.index(n) for n in spec.joint_names]]]
    resampled = resample(qpos, fps)
    # Upstream CSV interface uses root xyz, quaternion xyzw, selected joint order.
    csv = resampled[:, [0, 1, 2, 4, 5, 6, 3, *range(7, resampled.shape[1])]]
    return csv, dict(source=str(Path(path).resolve()), source_fps=fps, fps=50,
                     source_frames=len(qpos), frames=len(csv), robot=robot,
                     joint_names=list(spec.joint_names), quat_order='xyzw', up_axis='z',
                     sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def convert_corpus(input_dir, output_dir, robot, groups):
    root, output = Path(input_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for group in groups:
        paths = sorted((root / group).glob(f'*/{robot}/final/robot_motion.npz'))
        if not paths:
            raise FileNotFoundError(f'No {robot} motions below {root / group}')
        for path in paths:
            clip = path.parents[2].name
            name = f'{group}__{clip}'
            csv, metadata = convert_file(path, robot)
            target = output / (name + '.csv')
            temp = target.with_suffix('.tmp.csv')
            np.savetxt(temp, csv, delimiter=',', fmt='%.10g')
            temp.replace(target)
            records.append(dict(name=name, csv=str(target.resolve()), **metadata))
    manifest = dict(schema_version=1, robot=robot, fps=50, groups=list(groups), clips=records)
    temp = output / 'manifest.tmp.json'
    temp.write_text(json.dumps(manifest, indent=2) + '\n')
    temp.replace(output / 'manifest.json')
    return manifest
