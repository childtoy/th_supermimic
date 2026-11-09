#!/usr/bin/env python3
"""Evaluate frozen checkpoints and save matched primitive/transition review videos."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--backend', choices=['mjwarp', 'mujoco'], default='mjwarp')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--case', action='append', nargs=3, metavar=('LABEL', 'GROUP', 'CHECKPOINT'))
    args = parser.parse_args()
    os.environ['TH_SUPERMIMIC_ROBOT'] = 'k1'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    import torch
    from th_supermimic.data.playback import for_new_references
    from th_supermimic.sim.env import TrackingEnv
    from th_supermimic.rl.algos.supermimic.geometry import unflatten_state
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import load_policy
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import (
        flattened_qpos, reference_qpos, render_comparison,
    )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new empty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cases = args.case or [
        ('boxing_2000', 'boxing', 'runs/k1_boxing/ckpt_002000.pt'),
        ('boxing_12000', 'boxing', 'runs/k1_boxing/last.pt'),
        ('all_on_boxing', 'boxing', 'runs/k1_all/last.pt'),
        ('traffic1_12000', 'traffic_control1', 'runs/k1_traffic_control1/last.pt'),
        ('traffic2_12000', 'traffic_control2', 'runs/k1_traffic_control2/last.pt'),
    ]
    summaries = []
    for label, group, path in cases:
        if Path(label).name != label or label in {'.', '..'}:
            raise ValueError('Case label must be a directory name')
        output = args.output_dir / label
        output.mkdir()
        motion_dir = ROOT / 'data/k1' / group
        manifest = json.loads((motion_dir / 'manifest.json').read_text())
        files = [str(motion_dir / (c['name'] + '.csv')) for c in manifest['clips']]
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        torch.manual_seed(0)
        env = TrackingEnv(ROOT / 'assets/k1', files, len(files), backend=args.backend,
                          eval_mode=True, randomization=False, gamma=checkpoint['args']['gamma'])
        try:
            # Explicitly permit reference transfer while checking the full physics contract.
            inference = for_new_references(checkpoint, env.contract)
            policy = load_policy(env, inference)
            mdp, lib = env.supermimic_mdp, env.motion
            selected = [i for i, n in enumerate(lib.names) if 'primitive__' in n]
            transitions = [i for i, n in enumerate(lib.names) if 'primitive__' not in n]
            selected += [transitions[i] for i in sorted({0, len(transitions)//2, len(transitions)-1})] if transitions else []
            traces = {i: [] for i in selected}
            finished = torch.zeros(len(files), dtype=torch.bool, device=env.device)
            failed = finished.clone()
            returns = torch.zeros(len(files), device=env.device)
            steps = torch.zeros(len(files), dtype=torch.long, device=env.device)
            _, reset = env.reset_to_frames(lib.clip_start.clone())
            current = reset['supermimic_state']
            history = None
            with torch.inference_mode():
                initial = flattened_qpos(current['s']).cpu().numpy()
                for i in selected:
                    traces[i].append(initial[i].copy())
                while not bool(finished.all()):
                    obs = mdp.obs(unflatten_state(current['s']), current['a_prev'], current['xi'], current['f'])
                    hist = checkpoint['args']['hist']
                    history = obs[:, None].repeat(1, hist, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
                    action = policy(history, mdp.goal(current['f'], checkpoint['args']['goal_frames']))
                    state = env.step(action)
                    transition = state.info['supermimic']
                    alive = ~finished
                    actual = flattened_qpos(transition['s1']).cpu().numpy()
                    for i in selected:
                        if bool(alive[i]):
                            traces[i].append(actual[i].copy())
                    returns += torch.where(alive, transition['r'], 0.)
                    steps += alive
                    failed |= alive & transition['term']
                    finished |= alive & transition['done']
                    current = state.info['supermimic_state']
                scores = returns / ((lib.clip_len - 1).float() * mdp.max_step_reward)
                per_clip = {name: dict(norm_return=float(scores[i]), failed=bool(failed[i]),
                                      steps=int(steps[i]), frames=int(lib.clip_len[i]))
                            for i, name in enumerate(lib.names)}
                result = dict(label=label, group=group, checkpoint=str(Path(path).resolve()),
                              checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                              iteration=checkpoint['iteration'], backend=args.backend,
                              norm_return=float(scores.mean()), n_failed=int(failed.sum()),
                              n_clips=len(files), per_clip=per_clip,
                              training_contract=checkpoint['contract'], playback_contract=env.contract,
                              versions=dict(torch=torch.__version__))
                (output / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
                summaries.append({k: result[k] for k in ['label', 'norm_return', 'n_failed', 'n_clips']})
                print(json.dumps(summaries[-1]), flush=True)
                for i in selected:
                    name = lib.names[i]
                    ids = torch.arange(lib.clip_start[i], lib.clip_end[i] + 1, device=env.device)
                    reference = reference_qpos(lib, ids).cpu().numpy()
                    actual = np.asarray(traces[i])[:len(reference)]
                    completed = len(actual)
                    # Preserve the remaining target after failure; clearly label frozen policy pose.
                    actual = np.concatenate((actual, np.repeat(actual[-1:], len(reference)-completed, axis=0)))
                    row = per_clip[name]
                    labels = [f'{label} | {name} | return={row["norm_return"]:.3f}' +
                              (f' | FAILED at {row["steps"]/50:.2f}s (pose held)' if row['failed'] and t >= completed-1 else '')
                              for t in range(len(reference))]
                    np.savez_compressed(output / (name + '.npz'), reference=reference, policy=actual,
                                        fps=50, completed_frames=completed, failed=row['failed'])
                    render_comparison(reference, actual, output / (name + '.mp4'), ROOT / 'assets/k1/scene.xml',
                                      clip_name=name, width=args.width, height=args.height, fps=args.fps,
                                      global_ghost=True, frame_labels=labels)
                    print(f'Video: {label}/{name}', flush=True)
        finally:
            env.close()
    (args.output_dir / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    with (args.output_dir / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == '__main__':
    main()
