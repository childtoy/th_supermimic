#!/usr/bin/env python3
"""Evaluate archived policies and record raw actions, rewards and model errors."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['TH_SUPERMIMIC_ROBOT']='k1'
import numpy as np
import torch
from th_supermimic.sim.env import TrackingEnv
from th_supermimic.rl.algos.supermimic import networks
from th_supermimic.rl.algos.supermimic.geometry import unflatten_state
from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import load_policy
from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import flattened_qpos,reference_qpos


@torch.inference_mode()
def inspect(env,ckpt,mode,output):
    policy=load_policy(env,ckpt)
    args=ckpt['args'];mdp=env.supermimic_mdp;lib=env.motion
    wm=networks.WorldModel(hidden=args['wm_hidden'],num_joints=mdp.num_joints,context_dim=mdp.context_dim).to(env.device)
    wm.load_state_dict(ckpt['wm']);wm.eval()
    critic=None
    if 'critic' in ckpt and not args.get('no_critic',False):
        critic=networks.Critic(ckpt['critic']['norm.mean'].numel(),hidden=args['critic_hidden']).to(env.device)
        critic.load_state_dict(ckpt['critic']);critic.eval()
    env.reset_to_frames(lib.clip_start.clone())
    n=env.num_envs;dev=env.device
    finished=torch.zeros(n,dtype=torch.bool,device=dev)
    failed=finished.clone();returns=torch.zeros(n,device=dev);steps=torch.zeros(n,device=dev)
    totals={k:torch.zeros(n,device=dev) for k in ['track','action_penalty','joint_limit_penalty','joint_sq','root_m','relative_body_m','raw_abs','raw_oob','wm_mse','critic','critic_oob']}
    raw_max=torch.zeros(n,device=dev)
    critic_max=torch.full((n,),-float('inf'),device=dev)
    history=None;max_return=mdp.max_step_reward/(1-args['gamma'])
    selected=[i for i,name in enumerate(lib.names) if 'primitive__' in name]
    selected+=list(range(min(2,n)))
    selected=list(dict.fromkeys(selected))
    trace={i:dict(name=lib.names[i],policy=[],reference=[]) for i in selected}
    def stash(current):
        qpos=flattened_qpos(current['s']).cpu().numpy()
        ref=reference_qpos(lib,current['f']).cpu().numpy()
        for i in selected:
            if not finished[i]:trace[i]['policy'].append(qpos[i].copy());trace[i]['reference'].append(ref[i].copy())
    while not bool(finished.all()):
        current=env.state.info['supermimic_state'];state=unflatten_state(current['s'])
        obs=mdp.obs(state,current['a_prev'],current['xi'],current['f'])
        history=obs[:,None].repeat(1,args['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
        goal=mdp.goal(current['f'],args['goal_frames'])
        latent=torch.tanh(networks.run_mlp(policy.enc,policy.goal_norm(goal)))
        raw=networks.run_mlp(policy.dec,torch.cat((latent,policy.obs_norm(history).flatten(-2)),-1))
        action=raw.clamp(-20,20) if mode=='collector_bounds' else raw
        alive=~finished
        stash(current)
        raw_max=torch.where(alive,torch.maximum(raw_max,raw.abs().amax(-1)),raw_max)
        totals['raw_abs']+=torch.where(alive,raw.abs().mean(-1),0.)
        totals['raw_oob']+=torch.where(alive,(raw.abs()>20).float().mean(-1),0.)
        if critic is not None:
            inputs=torch.cat((mdp.critic_features(state,current['f'],mdp.normalize_context(current['xi'])),goal),-1)
            value=critic(inputs)
            totals['critic']+=torch.where(alive,value,0.)
            totals['critic_oob']+=torch.where(alive,(value>max_return).float(),0.)
            critic_max=torch.where(alive,torch.maximum(critic_max,value),critic_max)
        prediction=wm.step(state,action,mdp.normalize_context(current['xi']))
        result=env.step(action).info['supermimic']
        actual=unflatten_state(result['s1'])
        sigma=wm.out_norm.std().clamp(min=1e-3)
        wm_error=((wm.state_error(prediction,actual)/sigma)**2).mean(-1)
        totals['wm_mse']+=torch.where(alive,wm_error,0.)
        next_frames=torch.minimum(current['f']+1,lib.clip_end[lib.clip_id[current['f']]])
        _,terms=mdp.reward(actual,next_frames,action,current['a_prev'],terms=True)
        components=dict(track=terms['track']/mdp.max_step_reward,
            action_penalty=terms['action_rate']*mdp.cfg.w_action_rate*.02/mdp.max_step_reward,
            joint_limit_penalty=terms['joint_limit']*mdp.cfg.w_joint_limit*.02/mdp.max_step_reward,
            joint_sq=((actual[4]-lib.q[next_frames])**2).mean(-1),
            root_m=(actual[0]-lib.root_p[next_frames]).norm(dim=-1),
            relative_body_m=terms['e_body_pos'].sqrt())
        for key,value in components.items():totals[key]+=torch.where(alive,value,0.)
        returns+=torch.where(alive,result['r'],0.);steps+=alive
        failed|=alive&result['term'];finished|=alive&result['done']
    normal=returns/((lib.clip_len-1).float()*mdp.max_step_reward)
    per={}
    for i,name in enumerate(lib.names):
        row=dict(norm_return=float(normal[i]),failed=bool(failed[i]),steps=int(steps[i]),frames=int(lib.clip_len[i]),raw_max=float(raw_max[i]))
        row.update({k:float(v[i]/steps[i]) for k,v in totals.items()})
        row['joint_rmse']=row.pop('joint_sq')**.5
        if critic is not None:row['critic_max']=float(critic_max[i])
        per[name]=row
    summary=dict(mode=mode,iteration=ckpt['iteration'],clips=n,failed=int(failed.sum()),norm_return=float(normal.mean()),
        frame_weighted_return=float(returns.sum()/((lib.clip_len-1).sum()*mdp.max_step_reward)),
        theoretical_value_upper_bound=max_return,
        means={k:float((v/steps).mean()) for k,v in totals.items()},
        raw_max=float(raw_max.max()),critic_max=float(critic_max.max()) if critic is not None else None)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(summary=summary,per_clip=per),indent=2)+'\n')
    for i,t in trace.items():
        np.savez_compressed(output.with_name(output.stem+'__'+str(i)+'.npz'),name=t['name'],
            policy=np.asarray(t['policy']),reference=np.asarray(t['reference']),fps=50)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--groups',nargs='+',default=['boxing','traffic_control1','traffic_control2','all'])
    p.add_argument('--checkpoints',nargs='+',default=['ckpt_002000.pt','ckpt_004000.pt','last.pt'])
    p.add_argument('--modes',nargs='+',choices=['collector_bounds','original_unbounded_eval'],default=['collector_bounds'])
    p.add_argument('--backend',choices=['mjwarp','mujoco'],default='mjwarp')
    a=p.parse_args()
    torch.set_num_threads(1)
    for group in a.groups:
        motion=a.audit_root/'data/k1'/group
        manifest=json.loads((motion/'manifest.json').read_text())
        files=[str(motion/(c['name']+'.csv')) for c in manifest['clips']]
        env=TrackingEnv(a.audit_root/'assets/k1',files,len(files),backend=a.backend,eval_mode=True,randomization=False)
        try:
            for filename in a.checkpoints:
                c=torch.load(a.audit_root/'runs'/f'k1_{group}'/filename,map_location=env.device,weights_only=True)
                for mode in a.modes:
                    output=a.output_dir/f'{group}__{Path(filename).stem}__{mode}.json'
                    print(f'Inspecting {group} / {filename} / {mode}',flush=True)
                    inspect(env,c,mode,output)
                del c
        finally:env.close()
