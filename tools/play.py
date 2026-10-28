#!/usr/bin/env python3
"""Convert retargeted full sequences and evaluate/render a frozen policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', required=True, choices=['k1', 'g1'])
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--assets-dir', required=True, type=Path)
    p.add_argument('--input', required=True, type=Path, nargs='+', help='RIMKit final robot_motion.npz files')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--backend', default='mjwarp', choices=['mjwarp', 'mujoco'])
    p.add_argument('--video', nargs='*', default=[], help='Clip names to render; use all for every clip')
    a = p.parse_args()
    os.environ['TH_SUPERMIMIC_ROBOT'] = a.robot
    os.environ['TH_SUPERMIMIC_ASSETS'] = str(a.assets_dir.resolve())
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    import torch
    from th_supermimic.data.rmai import convert_file
    from th_supermimic.data.playback import for_new_references
    from th_supermimic.sim.env import TrackingEnv
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import evaluate
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import rollout_qpos, render_comparison

    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Use a new empty output directory to preserve previous results')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    converted = a.output_dir / 'references'
    converted.mkdir()
    records, files = [], []
    for path in a.input:
        name = path.parents[2].name if path.name == 'robot_motion.npz' else path.stem
        if name in [r['name'] for r in records]:
            raise ValueError(f'Duplicate clip name: {name}')
        values, metadata = convert_file(path, a.robot)
        csv = converted / (name + '.csv')
        np.savetxt(csv, values, delimiter=',', fmt='%.10g')
        records.append(dict(name=name, csv=str(csv.resolve()), **metadata))
        files.append(str(csv.resolve()))
    (converted / 'manifest.json').write_text(json.dumps(dict(schema_version=1, robot=a.robot, fps=50, clips=records), indent=2))
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    torch.manual_seed(0)
    env = TrackingEnv(a.assets_dir, files, num_envs=len(files), backend=a.backend,
                      eval_mode=True, randomization=False, gamma=checkpoint['args']['gamma'])
    try:
        inference = for_new_references(checkpoint, env.contract)
        selected = env.motion.names if 'all' in a.video else a.video
        if set(selected) - set(env.motion.names):
            raise ValueError('Unknown video clip name')
        result = evaluate(env, inference)
        result['checkpoint'] = str(a.checkpoint.resolve())
        result['checkpoint_sha256'] = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        result['training_contract'] = checkpoint['contract']
        result['playback_contract'] = env.contract
        result['reference_inputs'] = records
        (a.output_dir / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('norm_return', 'n_failed', 'per_clip')}, indent=2), flush=True)
        for name in selected:
            record = next(r for r in records if r['name'] == name)
            reference, policy = rollout_qpos(env, inference, name, seconds=record['frames'] / 50 + 1)
            np.savez_compressed(a.output_dir / (name + '.npz'), reference=reference, policy=policy, fps=50)
            render_comparison(reference, policy, a.output_dir / (name + '.mp4'),
                              a.assets_dir / 'scene.xml', clip_name=name, global_ghost=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
