#!/usr/bin/env python3
"""Evaluate phase subsets after training, preserving the physical contract."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True, help='Group directory containing phase subsets')
    p.add_argument('--assets-dir', type=Path, default=Path('assets/k1'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--expect-heldout', action='store_true')
    p.add_argument('--backend', choices=['mjwarp', 'mujoco'], default='mjwarp')
    a = p.parse_args()
    if a.output_dir.exists():
        p.error('Use a new evaluation output directory')
    os.environ['TH_SUPERMIMIC_ROBOT'] = 'k1'
    os.environ['TH_SUPERMIMIC_ASSETS'] = str(a.assets_dir.resolve())
    import torch
    import mujoco
    from th_supermimic.sim.env import TrackingEnv
    from th_supermimic.data.playback import for_new_references
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import evaluate

    torch.set_num_threads(1)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    trained = {m['name'] for m in checkpoint['contract']['motions']}
    plans = []
    for subset in ['interpolation', 'extrapolation']:
        directory = a.data_dir/subset
        manifest = json.loads((directory/'manifest.json').read_text())
        files = [directory/(c['name']+'.csv') for c in manifest['clips']]
        overlap = sorted({f.name for f in files} & trained)
        if a.expect_heldout and overlap:
            raise ValueError(f'Training leakage in {subset}: {len(overlap)} clips')
        plans.append((subset, files, overlap))
    a.output_dir.mkdir(parents=True)
    summary = {}
    for subset, files, overlap in plans:
        if not files:
            summary[subset] = dict(n_clips=0, skipped=True)
            continue
        torch.manual_seed(checkpoint['args']['seed'])
        env = TrackingEnv(a.assets_dir, [str(f) for f in files], len(files), backend=a.backend,
                          eval_mode=True, randomization=False, gamma=checkpoint['args']['gamma'])
        try:
            result = evaluate(env, for_new_references(checkpoint, env.contract))
            result.update(checkpoint=str(a.checkpoint.resolve()), iteration=checkpoint['iteration'],
                          checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                          subset=subset, n_clips=len(files), seen_training_clips=overlap,
                          training_contract=checkpoint['contract'], evaluation_contract=env.contract,
                          backend=a.backend, versions=dict(torch=torch.__version__, mujoco=mujoco.__version__))
            (a.output_dir/(subset+'.json')).write_text(json.dumps(result, indent=2)+'\n')
            with (a.output_dir/(subset+'.csv')).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=['clip', 'norm_return', 'steps', 'failed'])
                writer.writeheader()
                for name, row in result['per_clip'].items():
                    writer.writerow(dict(clip=name, **row))
            summary[subset] = dict(n_clips=len(files), norm_return=result['norm_return'],
                                   n_failed=result['n_failed'], seen_training_clips=len(overlap))
            print(json.dumps(dict(subset=subset, **summary[subset])), flush=True)
        finally:
            env.close()
    (a.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
