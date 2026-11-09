#!/usr/bin/env python3
"""Offline diagnostic: compare closed-loop WM bridge scores to real physics.

Rollouts start from simulator-observed state/history. This is an oracle-start
calibration, not a simulator-free sampler or an interval success certificate.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def correlations(rows):
    import numpy as np
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a)>1 and np.std(a)>1e-9 and np.std(b)>1e-9 else None
    def ranks(a):
        return np.array([sum(x<v for x in a)+(sum(x==v for x in a)-1)/2 for v in a])
    a = np.array([r['wm_return'] for r in rows])
    b = np.array([r['sim_return'] for r in rows])
    return dict(n=len(rows), pearson=corr(a,b), spearman=corr(ranks(a),ranks(b)),
                mean_absolute_score_error=float(np.abs(a-b).mean()) if len(a) else None,
                mean_wm_return=float(a.mean()) if len(a) else None,
                mean_sim_return=float(b.mean()) if len(b) else None,
                wm_predicted_terminations=sum(r['wm_failed'] for r in rows),
                sim_terminations=sum(r['sim_failed'] for r in rows),
                thresholds=[dict(threshold=t,
                    predicted_pass=sum(r['wm_return']>=t and not r['wm_failed'] for r in rows),
                    false_pass=sum(r['wm_return']>=t and not r['wm_failed'] and
                                   (r['sim_return']<t or r['sim_failed']) for r in rows),
                    false_reject=sum((r['wm_return']<t or r['wm_failed']) and
                                     r['sim_return']>=t and not r['sim_failed'] for r in rows))
                    for t in [.6,.7,.8,.9]])


def inspect(env, checkpoint, records, primitives):
    import torch
    from th_supermimic.data.playback import for_new_references
    from th_supermimic.rl.algos.supermimic import networks
    from th_supermimic.rl.algos.supermimic.geometry import unflatten_state
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import load_policy
    mdp, lib, args = env.supermimic_mdp, env.motion, checkpoint['args']
    policy = load_policy(env, for_new_references(checkpoint, env.contract))
    wm = networks.WorldModel(hidden=args['wm_hidden'], num_joints=mdp.num_joints,
                              context_dim=mdp.context_dim).to(env.device)
    wm.load_state_dict(checkpoint['wm']); wm.eval()
    bounds = []
    for record in records:
        match = re.search(r'([^_]+(?:_[^_]+)*)__to__([^_]+(?:_[^_]+)*)__a(\d+)__c(\d+)$', record['name'])
        if not match or int(match[4]):
            raise ValueError('Need entry-mode transition clips')
        source, target, stop, _ = match.groups()
        group = record['name'].split('__')[0]
        bridge = record['source_frames']-int(stop)-primitives[(group,target)]['source_frames']
        if bridge<1:
            raise ValueError('Invalid bridge duration')
        # Start immediately before the first bridge frame.
        lo = round(int(stop)*50/30)-1
        hi = round((int(stop)+bridge)*50/30)-1
        bounds.append((lo,hi))
    n = len(records)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    _, reset = env.reset_to_frames(lib.clip_start.clone())
    current = reset['supermimic_state']; history = None
    states = [current['s'].clone()]
    histories, previous, actions, rewards, alive, terms = [], [], [], [], [], []
    context = current['xi'].clone()
    while not bool(finished.all()):
        obs = mdp.obs(unflatten_state(current['s']),current['a_prev'],current['xi'],current['f'])
        history = obs[:,None].repeat(1,args['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
        histories.append(history.clone()); previous.append(current['a_prev'].clone())
        action = policy(history,mdp.goal(current['f'],args['goal_frames']))
        alive.append((~finished).clone())
        transition = env.step(action).info['supermimic']
        states.append(transition['s1'].clone()); actions.append(action)
        rewards.append(transition['r']); terms.append(transition['term'] & ~finished)
        finished |= transition['done']
        current = env.state.info['supermimic_state']
    states, histories, previous, actions, rewards, alive, terms = [torch.stack(x) for x in
        [states,histories,previous,actions,rewards,alive,terms]]
    total = len(rewards)
    rows = []
    # A 32-step prefix, two re-anchored windows and a free rollout over the full bridge.
    for kind, offset, cap in [('prefix32',0,32),('middle32',32,32),('tail32',64,32),('full_bridge',0,None)]:
        valid = [i for i,(lo,hi) in enumerate(bounds) if lo+offset<hi and lo+offset<total and bool(alive[lo+offset,i])]
        if not valid:
            continue
        ids = torch.tensor(valid, device=env.device)
        times = torch.tensor([bounds[i][0]+offset for i in valid], device=env.device)
        lengths = torch.tensor([min(cap or bounds[i][1]-bounds[i][0],bounds[i][1]-bounds[i][0]-offset) for i in valid],device=env.device)
        state = unflatten_state(states[times,ids])
        hist = histories[times,ids].clone(); prev = previous[times,ids]
        xi = context[ids]; normalized = mdp.normalize_context(xi)
        frames = lib.clip_start[ids]+times
        count = len(valid)
        predicted_alive = torch.ones(count,dtype=torch.bool,device=env.device)
        accum = {k:torch.zeros(count,device=env.device) for k in
                 ['wm_reward','sim_reward','wm_q_sq','sim_q_sq','q_pred_sq','root_pred_sq','one_step_q_sq','wm_steps','sim_steps','common_steps']}
        wm_fail = torch.zeros(count,dtype=torch.bool,device=env.device); sim_fail=wm_fail.clone()
        for k in range(int(lengths.max())):
            within = k<lengths
            action = policy(hist,mdp.goal(frames,args['goal_frames']))
            prediction = wm.step(state,action,normalized)
            next_frames = torch.minimum(frames+1,lib.clip_end[ids])
            reward = mdp.reward(prediction,next_frames,action,prev)
            failed = mdp.termination(prediction,next_frames)
            pred_mask = within & predicted_alive
            t = (times+k).clamp(max=total-1)
            actual_mask = within & (times+k<total) & alive[t,ids]
            common = actual_mask & pred_mask
            actual = unflatten_state(states[t+1,ids])
            one_step = wm.step(unflatten_state(states[t,ids]),actions[t,ids],normalized)
            values = dict(wm_reward=torch.where(pred_mask,reward,0.),
                          sim_reward=torch.where(actual_mask,rewards[t,ids],0.),
                          wm_q_sq=torch.where(pred_mask,(prediction[4]-lib.q[next_frames]).square().mean(-1),0.),
                          sim_q_sq=torch.where(actual_mask,(actual[4]-lib.q[next_frames]).square().mean(-1),0.),
                          q_pred_sq=torch.where(common,(prediction[4]-actual[4]).square().mean(-1),0.),
                          root_pred_sq=torch.where(common,(prediction[0]-actual[0]).square().sum(-1),0.),
                          one_step_q_sq=torch.where(actual_mask,(one_step[4]-actual[4]).square().mean(-1),0.),
                          wm_steps=pred_mask,sim_steps=actual_mask,common_steps=common)
            for key,value in values.items():
                accum[key] += value
            wm_fail |= pred_mask & failed; sim_fail |= actual_mask & terms[t,ids]
            predicted_alive &= ~failed
            new_obs = mdp.obs(prediction,action,xi,next_frames)
            hist = torch.cat((hist[:,1:],new_obs[:,None]),1)
            state,prev,frames = prediction,action,next_frames
        for j,i in enumerate(valid):
            a = {key:float(value[j]) for key,value in accum.items()}
            rows.append(dict(clip=records[i]['name'],kind=kind,horizon_steps=int(lengths[j]),
                wm_return=a['wm_reward']/int(lengths[j])/mdp.max_step_reward,
                sim_return=a['sim_reward']/int(lengths[j])/mdp.max_step_reward,
                wm_failed=bool(wm_fail[j]),sim_failed=bool(sim_fail[j]),
                wm_tracking_joint_rmse=(a['wm_q_sq']/max(1,a['wm_steps']))**.5,
                sim_tracking_joint_rmse=(a['sim_q_sq']/max(1,a['sim_steps']))**.5,
                multistep_prediction_joint_rmse=(a['q_pred_sq']/max(1,a['common_steps']))**.5,
                multistep_prediction_root_rmse_m=(a['root_pred_sq']/max(1,a['common_steps']))**.5,
                one_step_prediction_joint_rmse=(a['one_step_q_sq']/max(1,a['sim_steps']))**.5,
                sim_observed_steps=int(a['sim_steps'])))
    summary = {kind:correlations([r for r in rows if r['kind']==kind])
               for kind in ['prefix32','middle32','tail32','full_bridge']}
    summary['prefix_failed_before_bridge'] = sum(not bool(alive[lo,i]) if lo<total else True for i,(lo,_) in enumerate(bounds))
    summary['sim_full_clip_failures'] = int(terms.any(0).sum())
    summary['full_clip_return'] = float((torch.where(alive,rewards,0.).sum(0)/((lib.clip_len-1)*mdp.max_step_reward)).mean())
    return dict(summary=summary,windows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--group',required=True)
    p.add_argument('--checkpoints',nargs='+',default=['ckpt_000600.pt','ckpt_001400.pt','last.pt'])
    p.add_argument('--subset',default='interpolation',choices=['interpolation','extrapolation'])
    p.add_argument('--assets-dir',type=Path,default=Path('assets/k1'))
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--backend',default='mjwarp',choices=['mjwarp','mujoco'])
    a = p.parse_args()
    if a.output_dir.exists():
        p.error('Use a new output directory')
    os.environ['TH_SUPERMIMIC_ROBOT']='k1'
    import torch
    from th_supermimic.sim.env import TrackingEnv
    torch.set_num_threads(1)
    directory=a.run_dir/'data'/a.group
    records=json.loads((directory/a.subset/'manifest.json').read_text())['clips']
    dense=json.loads((directory/'dense/manifest.json').read_text())['clips']
    primitives={(c['name'].split('__')[0],c['name'].split('__primitive__')[1]):c for c in dense if '__primitive__' in c['name']}
    files=[str(directory/a.subset/(r['name']+'.csv')) for r in records]
    a.output_dir.mkdir(parents=True)
    all_rows=[]
    for filename in a.checkpoints:
        path=a.run_dir/('k1_'+a.group)/filename
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        trained={m['name'] for m in checkpoint['contract']['motions']}
        if trained & {Path(f).name for f in files}:
            raise ValueError('Heldout reference appears in training contract')
        torch.manual_seed(checkpoint['args']['seed'])
        env=TrackingEnv(a.assets_dir,files,len(files),backend=a.backend,eval_mode=True,randomization=False)
        try:
            with torch.inference_mode():
                result=inspect(env,checkpoint,records,primitives)
            result.update(checkpoint=str(path.resolve()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          iteration=checkpoint['iteration'],subset=a.subset,
                          protocol='Oracle simulator state/history starts; closed-loop policy+WM; no critic; exploratory thresholds only',
                          training_contract=checkpoint['contract'],evaluation_contract=env.contract)
            (a.output_dir/(path.stem+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            for row in result['windows']:
                all_rows.append(dict(checkpoint=path.name,iteration=checkpoint['iteration'],**row))
            print(json.dumps(dict(checkpoint=path.name,summary=result['summary'])),flush=True)
        finally:
            env.close()
    with (a.output_dir/'windows.csv').open('w') as stream:
        fields=list(all_rows[0]) if all_rows else ['checkpoint','iteration','clip','kind','horizon_steps','wm_return','sim_return']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(all_rows)
    (a.output_dir/'pooled_summary.json').write_text(json.dumps({kind:correlations([r for r in all_rows if r['kind']==kind])
        for kind in ['prefix32','middle32','tail32','full_bridge']},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
