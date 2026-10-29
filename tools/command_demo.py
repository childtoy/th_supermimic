#!/usr/bin/env python3
"""Frozen-policy, immediate reference switching from a bank of trained pair clips."""
import argparse
import json
import os
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', choices=['k1', 'g1'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--motion-dir', type=Path, required=True)
    p.add_argument('--assets-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--switches', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--source', help='Fixed single-switch source primitive')
    p.add_argument('--target', help='Fixed single-switch target primitive')
    p.add_argument('--cutback-seconds', type=float, help='Cut source this many seconds before its end; requires an exact bank edge')
    a = p.parse_args()
    fixed = any(x is not None for x in (a.source, a.target, a.cutback_seconds))
    if fixed and (a.source is None or a.target is None or a.cutback_seconds is None or a.switches != 1):
        p.error('Fixed playback requires --source, --target, --cutback-seconds and --switches 1')
    if a.switches < 1:
        p.error('--switches must be positive')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Use a new empty output directory')
    os.environ['TH_SUPERMIMIC_ROBOT'] = a.robot
    os.environ['TH_SUPERMIMIC_ASSETS'] = str(a.assets_dir.resolve())
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    import torch
    from th_supermimic.sim.env import TrackingEnv
    from th_supermimic.rl.algos.supermimic.geometry import unflatten_state, yaw_mat
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import load_policy
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import flattened_qpos, reference_qpos, render_comparison

    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.motion_dir / 'manifest.json').read_text())
    records = manifest['clips']
    files = [str((a.motion_dir / (r['name'] + '.csv')).resolve()) for r in records]
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    env = TrackingEnv(a.assets_dir, files, num_envs=1, backend='mjwarp', eval_mode=True,
                      randomization=False, gamma=checkpoint['args']['gamma'])
    try:
        lib, mdp = env.motion, env.supermimic_mdp
        policy = load_policy(env, checkpoint)  # Exact original training bank required.
        primitives = {}
        edges = []
        for i, r in enumerate(records):
            if '__primitive__' in r['name']:
                primitives[r['name'].split('__primitive__')[1]] = i
        for i, r in enumerate(records):
            match = re.search(r'([^_]+(?:_[^_]+)*)__to__([^_]+(?:_[^_]+)*)__a(\d+)__c(\d+)$', r['name'])
            if not match:
                continue
            source, target, stop, start = match.groups()
            if int(start) != 0:
                continue
            # Corpus stores A[:stop] + bridge + full C; retain its measured timing.
            target_frames = records[primitives[target]]['source_frames']
            bridge = r['source_frames'] - int(stop) - target_frames
            if bridge < 1:
                raise ValueError('Invalid bridge length inferred from corpus metadata')
            edges.append(dict(index=i, source=source, target=target, stop=int(stop), bridge=bridge))
        if len(primitives) < 2 or not edges:
            raise ValueError('Need primitive and entry-mode pair clips')
        # Immutable local bank. Transform chosen clip into the current reference world frame.
        fields = ('root_p', 'root_R', 'root_v', 'root_w', 'body_p', 'body_R', 'body_v', 'body_w')
        original = {key: getattr(lib, key).clone() for key in fields}
        active = a.source if fixed else sorted(primitives)[0]
        fixed_edge = None
        if fixed:
            if active not in primitives or a.target not in primitives or active == a.target:
                raise ValueError('Choose two different known primitives')
            source_frames = records[primitives[active]]['source_frames']
            if not np.isfinite(a.cutback_seconds) or a.cutback_seconds < 0:
                raise ValueError('Cutback must be finite and nonnegative')
            stop = source_frames - round(a.cutback_seconds * 30)
            if stop < 10:
                raise ValueError('Cutback is outside source length/context bounds')
            fixed_edge = next((e for e in edges if e['source'] == active and e['target'] == a.target and e['stop'] == stop), None)
            if fixed_edge is None:
                raise ValueError('No exact cutback edge exists in the bank')
        initial = int(lib.clip_start[primitives[active]])
        env.reset_to_frames(torch.tensor([initial], device=env.device))
        target_start = initial
        # Commands are generated online after varying amounts of each primitive.
        next_phase = (fixed_edge['stop'] - 1) / 30 if fixed else rng.uniform(.55, 1.15)
        history = None
        events, refs, poses, labels, rewards, errors = [], [], [], [], [], []
        status = 'time_limit'
        post_last = None
        with torch.inference_mode():
            for step in range(50 * 90):
                current = env.state.info['supermimic_state']
                f = int(current['f'][0])
                phase = (f - target_start) / 50
                if len(events) < a.switches and phase >= next_phase:
                    target = a.target if fixed else rng.choice(sorted(set(primitives) - {active}))
                    candidates = [e for e in edges if e['source'] == active and e['target'] == target]
                    if not candidates:
                        raise ValueError(f'No edge {active} -> {target}')
                    edge = fixed_edge if fixed else min(candidates, key=lambda e: abs((e['stop'] - 1) / 30 - phase))
                    index = edge['index']
                    lo, hi = int(lib.clip_start[index]), int(lib.clip_end[index]) + 1
                    # Anchor on the last source frame; the very next step enters the bridge.
                    anchor = lo + round((edge['stop'] - 1) * 50 / 30)
                    old_p, old_R = lib.root_p[f].clone(), lib.root_R[f].clone()
                    rotation = yaw_mat(old_R) @ yaw_mat(original['root_R'][anchor]).T
                    offset = old_p - rotation @ original['root_p'][anchor]
                    offset[2] = 0  # Only XY translation and yaw, preserving reference heights.
                    for key in fields:
                        x = original[key][lo:hi]
                        transformed = rotation @ x if key.endswith('_R') else (rotation @ x[..., None]).squeeze(-1)
                        if key.endswith('_p'):
                            transformed = transformed + offset
                        getattr(lib, key)[lo:hi] = transformed
                    ref_jump = float((lib.q[anchor] - lib.q[f]).square().mean().sqrt())
                    env.f[0] = anchor  # No sim reset, history reset, action reset, or crossfade.
                    env._publish()
                    target_start = lo + round((edge['stop'] + edge['bridge']) * 50 / 30)
                    events.append(dict(command=len(events) + 1, time_s=step / 50, source=active, target=target,
                                       source_phase_s=phase, selected_source_phase_s=(edge['stop'] - 1) / 30,
                                       phase_error_s=(edge['stop'] - 1) / 30 - phase,
                                       edge=records[index]['name'], bridge_s=edge['bridge'] / 30,
                                       reference_joint_jump_rmse_rad=ref_jump, response_delay_s=0.0))
                    active = target
                    next_phase = rng.uniform(.55, 1.15)
                    current = env.state.info['supermimic_state']
                    f = int(current['f'][0])
                    if len(events) == a.switches:
                        post_last = target_start + 60
                refs.append(reference_qpos(lib, current['f'])[0].cpu().numpy().copy())
                poses.append(flattened_qpos(current['s'])[0].cpu().numpy().copy())
                stage = 'transition' if f < target_start else 'playing'
                labels.append(f"cmd {len(events)}/{a.switches}: {active} ({stage})")
                errors.append(float((unflatten_state(current['s'])[4] - lib.q[current['f']]).square().mean().sqrt()))
                obs = mdp.obs(unflatten_state(current['s']), current['a_prev'], current['xi'], current['f'])
                history = obs[:, None].repeat(1, checkpoint['args']['hist'], 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
                action = policy(history, mdp.goal(current['f'], checkpoint['args']['goal_frames']))
                transition = env.step(action).info['supermimic']
                rewards.append(float(transition['r'][0]))
                if bool(transition['done'][0]):
                    # Record actual pre-reset terminal pose, never an auto-reset pose.
                    poses.append(flattened_qpos(transition['s1'])[0].cpu().numpy().copy())
                    refs.append(reference_qpos(lib, torch.tensor([min(f+1, int(lib.clip_end[lib.clip_id[f]]))], device=env.device))[0].cpu().numpy().copy())
                    labels.append('TERMINATED' if bool(transition['term'][0]) else 'REFERENCE END')
                    status = 'failed' if bool(transition['term'][0]) else 'reference_ended'
                    break
                if post_last is not None and f >= post_last:
                    status = 'completed'
                    break
        result = dict(status=status, commands_issued=len(events), requested_commands=a.switches,
                      duration_s=(len(poses)-1)/50, normalized_step_reward=sum(rewards)/len(rewards)/mdp.max_step_reward,
                      mean_joint_rmse_rad=sum(errors)/len(errors), events=events,
                      checkpoint=str(a.checkpoint.resolve()), seed=a.seed, cutback_seconds=a.cutback_seconds,
                      method='Immediate nearest-phase pair selection; planar reference alignment; persistent history and physical state. No blending or retraining.',
                      limitation='Commands sampled during primitive playback, not during a bridge; no arbitrary rapid-command guarantee.')
        (a.output_dir / 'commands.json').write_text(json.dumps(result, indent=2) + '\n')
        np.savez_compressed(a.output_dir / 'rollout.npz', reference=np.array(refs), policy=np.array(poses), labels=np.array(labels), fps=50)
        print(json.dumps(result, indent=2), flush=True)
        render_comparison(np.array(refs), np.array(poses), a.output_dir / 'commands.mp4', a.assets_dir / 'scene.xml',
                          clip_name='Immediate reference switching', global_ghost=True, frame_labels=labels)
    finally:
        env.close()


if __name__ == '__main__':
    main()
