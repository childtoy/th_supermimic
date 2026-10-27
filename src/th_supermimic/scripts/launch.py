"""Standalone CLI; choose robot before importing the tracking mathematics."""
from __future__ import annotations
import argparse
from dataclasses import asdict,fields
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',choices=['k1','g1'],required=True)
    p.add_argument('--motion-dir',type=Path,required=True)
    p.add_argument('--assets-dir',type=Path)
    p.add_argument('--backend',choices=['mjwarp','mujoco'],default='mjwarp')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--iterations',type=int)
    p.add_argument('--config',type=Path,help='JSON containing SuperMimicConfig fields')
    p.add_argument('--resume',type=Path)
    p.add_argument('--eval',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--no-compile',action='store_true')
    p.add_argument('--no-randomization',action='store_true')
    p.add_argument('--config-only',action='store_true')
    a,overrides=p.parse_known_args()
    assets=(a.assets_dir or Path('assets')/a.robot).resolve()
    os.environ['TH_SUPERMIMIC_ROBOT']=a.robot
    os.environ['TH_SUPERMIMIC_ASSETS']=str(assets)
    manifest=json.loads((a.motion_dir/'manifest.json').read_text())
    if manifest['robot']!=a.robot or manifest['fps']!=50:
        raise ValueError('Imported manifest robot/fps mismatch')
    files=[(a.motion_dir/(clip['name']+'.csv')).resolve() for clip in manifest['clips']]
    if not files or any(not x.is_file() for x in files):
        raise FileNotFoundError('Missing CSV clips; run tools/convert_rmai.py')
    metadata=json.loads((assets/'metadata.json').read_text())
    if metadata['robot']!=a.robot:
        raise ValueError('Asset robot mismatch')
    import torch
    from th_supermimic.robots import ROBOT
    if ROBOT.name != a.robot:
        raise RuntimeError('Robot-dependent modules were imported before robot selection; start a new process')
    from th_supermimic.rl.algos.supermimic.learner import SuperMimicConfig,SuperMimicLearner
    from th_supermimic.sim.env import TrackingEnv
    checkpoint=a.resume or (a.checkpoint or a.run_dir/'last.pt' if a.eval else None)
    values={'iterations':12000}
    if checkpoint:
        values.update(torch.load(checkpoint,map_location='cpu',weights_only=True)['args'])
    if a.config:
        values.update(json.loads(a.config.read_text()))
    if a.smoke:
        values.update(iterations=3,buffer_steps=64,warmup_steps=16,wm_only_iters=0,wm_window=2,
            horizon=4,wm_batches=1,wm_batch=8,pi_batch=8,wm_hidden=(64,64),policy_hidden=(64,64),
            critic_hidden=(64,32),critic_epochs=1,critic_minibatches=1,compile=False,amp='none',
            ckpt_every=1,log_every=1)
        a.num_envs=8
    valid={f.name for f in fields(SuperMimicConfig)}
    for override in overrides:
        key,sep,raw=override.partition('=')
        key=key.removeprefix('algo.')
        if not sep or key not in valid:
            p.error(f'Unknown option {override}; use --eval/--resume for checkpoint operations')
        try: value=json.loads(raw)
        except json.JSONDecodeError: value=raw
        values[key]=value
    if a.iterations is not None:
        values['iterations']=a.iterations
    if a.no_compile:
        values['compile']=False
    if set(values)-valid:
        raise ValueError(f'Unknown configuration fields: {set(values)-valid}')
    for key in ('wm_hidden','policy_hidden','critic_hidden','test_hidden_dims'):
        if values.get(key) is not None:
            values[key]=tuple(values[key])
    cfg=SuperMimicConfig(**values)
    if cfg.iterations<1 or a.num_envs<1 or not 0<cfg.gamma<1:
        p.error('iterations/env count must be positive and 0 < gamma < 1')
    snapshot=dict(schema=2,robot=a.robot,backend=a.backend,assets_dir=str(assets),
                  motion_files=[str(x) for x in files],learner=asdict(cfg))
    if a.config_only:
        print(json.dumps(snapshot,indent=2));return
    if not a.resume and not a.eval and (a.run_dir/'last.pt').exists():
        raise FileExistsError('Run already has a checkpoint; use --resume or a fresh run directory')
    torch.manual_seed(cfg.seed)
    env=TrackingEnv(assets,[str(x) for x in files],num_envs=len(files) if a.eval else a.num_envs,
                    backend=a.backend,eval_mode=a.eval,gamma=cfg.gamma,seed=cfg.seed,
                    randomization=not a.no_randomization)
    try:
        if a.eval:
            from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import evaluate
            saved=torch.load(checkpoint,map_location=env.device,weights_only=True)
            result=evaluate(env,saved)
            destination=Path(checkpoint).with_name(Path(checkpoint).stem+'_eval.json')
            destination.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2));return
        a.run_dir.mkdir(parents=True,exist_ok=True)
        (a.run_dir/'run_config.json').write_text(json.dumps(snapshot,indent=2)+'\n')
        learner=SuperMimicLearner(env,cfg,a.run_dir,device=env.device,backend=a.backend)
        print(json.dumps(learner.train(resume=a.resume),indent=2))
    finally:
        env.close()


def evaluate_main():
    sys.argv.append('--eval')
    main()


def video_main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--robot',choices=['k1','g1'],required=True)
    a,rest=p.parse_known_args()
    os.environ['TH_SUPERMIMIC_ROBOT']=a.robot
    os.environ['TH_SUPERMIMIC_ASSETS']=str((Path('assets')/a.robot).resolve())
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import main as render
    render(rest)


if __name__=='__main__':
    main()
