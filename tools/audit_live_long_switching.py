#!/usr/bin/env python3
"""Independent live A->bridge->B audit, with no source-state cache restoration.

Runs policy, zero-after-command and direct-cut controls from A frame zero.
Logs all simulator reset calls and verifies that reference handoffs leave
physical state, previous actions and observation history unchanged.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['TH_SUPERMIMIC_ROBOT']='k1'


def json_scalar(value):
    """Serialize NumPy scalars while rejecting unsupported objects."""
    import numpy as np
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(f'Unsupported result type: {type(value).__name__}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dirs',nargs='+',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--base-dir',type=Path,default=Path('runs/long_full_base_20261103'))
    p.add_argument('--modes',nargs='+',choices=['policy','zero_after_command','direct_cut'],default=['policy','zero_after_command','direct_cut'])
    a=p.parse_args()
    if a.output_dir.exists():
        p.error('Use a new output directory')
    import numpy as np
    import torch
    from th_supermimic.sim.env import TrackingEnv
    from th_supermimic.data.playback import for_new_references
    from th_supermimic.data.rmai import convert_file
    from th_supermimic.rl.algos.supermimic import networks
    from th_supermimic.rl.algos.supermimic.geometry import unflatten_state,yaw_mat
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import load_policy
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import flattened_qpos,reference_qpos
    from calibrate_transition_wm import correlations
    torch.set_num_threads(1);torch.manual_seed(0)
    checkpoint_path=a.base_dir/'train/best_coverage.pt'
    digest=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    candidates=[]
    for directory in a.case_dirs:
        bundle=json.loads((directory/'cases.json').read_text())
        if bundle['checkpoint_sha256']!=digest:
            raise ValueError('Candidate bundle used a different base checkpoint')
        for c in bundle['cases']:
            candidates.append(dict(c,directory=str(directory)))
    if len({c['name'] for c in candidates})!=len(candidates):
        raise ValueError('Duplicate candidate')
    a.output_dir.mkdir(parents=True)
    refs=a.output_dir/'references';refs.mkdir()
    base_manifest=json.loads((a.base_dir/'data/manifest.json').read_text())
    groups=[r['name'].split('__')[0] for r in base_manifest['clips']]
    files=[str(a.base_dir/'data'/(r['name']+'.csv')) for r in base_manifest['clips']]
    source_indices=dict(zip(groups,range(len(groups))))
    base_csv=dict(zip(groups,files))
    bridge_indices=[]
    for c in candidates:
        native=Path(c['directory'])/'robots_mj360'/c['source']/c['name']/'k1/final/robot_motion.npz'
        values,_=convert_file(native,'k1')
        path=refs/('bridge__'+c['name']+'.csv');np.savetxt(path,values,delimiter=',',fmt='%.10g')
        bridge_indices.append(len(files));files.append(str(path))
    trials=[]
    for ci,c in enumerate(candidates):
        for mode in a.modes:
            target_path=refs/(f'target__{len(trials):04d}__'+c['target']+'.csv')
            target_path.write_bytes(Path(base_csv[c['target']]).read_bytes())
            target_index=len(files);files.append(str(target_path))
            trials.append(dict(case=c['name'],source=c['source'],target=c['target'],phase_s=c['phase_s'],
                               mode=mode,candidate_index=ci,target_index=target_index))
    trial_count=len(trials)
    env=TrackingEnv('assets/k1',files,trial_count,backend='mjwarp',eval_mode=True,randomization=False)
    try:
        policy=load_policy(env,for_new_references(checkpoint,env.contract));lib=env.motion;mdp=env.supermimic_mdp
        device=env.device
        initial=torch.tensor([int(lib.clip_start[source_indices[t['source']]]) for t in trials],device=device)
        env.reset_to_frames(initial)
        history=None
        done=torch.zeros(trial_count,dtype=torch.bool,device=device)
        stage=np.zeros(trial_count,np.int8)
        command_steps=np.array([round(t['phase_s']*50) for t in trials])
        command_at=np.full(trial_count,-1,int);handoff_at=np.full(trial_count,-1,int)
        final_at=np.full(trial_count,-1,int)
        completed=np.zeros(trial_count,bool);failed=np.zeros(trial_count,bool)
        reset_counts=np.zeros(trial_count,int);unexpected_resets=[];events=[]
        reset_ids=[]
        original_reset=env.sim.reset
        def counted_reset(ids,qpos,qvel):
            reset_ids.extend(ids.detach().cpu().tolist())
            return original_reset(ids,qpos,qvel)
        env.sim.reset=counted_reset
        aligned=set()
        H=127;C=len(candidates);F=2*C
        D=env.state.info['supermimic_state']['s'].shape[-1]
        J=mdp.num_joints;O=policy.obs_norm.mean.numel()
        wm_initial=torch.zeros(F,D,device=device)
        wm_history=torch.zeros(F,checkpoint['args']['hist'],O,device=device)
        wm_prev=torch.zeros(F,J,device=device);wm_xi=torch.zeros(F,mdp.context_dim,device=device)
        wm_anchors=torch.zeros(F,dtype=torch.long,device=device);wm_available=np.zeros(F,bool)
        actual_states=torch.zeros(H+1,F,D,device=device)
        actual_actions=torch.zeros(H,F,J,device=device)
        actual_rewards=torch.zeros(H,F,device=device)
        actual_alive=torch.zeros(H,F,dtype=torch.bool,device=device)
        actual_failed=torch.zeros(H,F,dtype=torch.bool,device=device)
        sums={key:np.zeros(trial_count) for key in ['transition_reward','transition_joint_sq','transition_body_sq',
               'transition_root_sq','transition_body_ori_sq','transition_action_rate','target_reward','target_joint_sq',
               'target_body_sq','target_root_sq','target_body_ori_sq','target_count','transition_count']}
        peaks=np.zeros(trial_count);boundaries=[{} for _ in trials]
        video_indices=[i for i,t in enumerate(trials) if t['phase_s'] in [5.,15.] and
                       (t['source'],t['target']) in [('boxing','traffic_control1'),('traffic_control1','traffic_control2')] and
                       t['mode'] in ['policy','direct_cut']][:8]
        video_pose=[];video_ref=[];video_alive=[];video_stage=[]

        def align(index,anchor,old_frame):
            lo=int(lib.clip_start[index]);hi=int(lib.clip_end[index])+1
            rotation=yaw_mat(lib.root_R[old_frame])@yaw_mat(lib.root_R[anchor]).T
            offset=lib.root_p[old_frame]-rotation@lib.root_p[anchor];offset[2]=0
            for key in ['root_p','root_R','root_v','root_w','body_p','body_R','body_v','body_w']:
                x=getattr(lib,key)[lo:hi].clone()
                y=rotation@x if key.endswith('_R') else (rotation@x[...,None]).squeeze(-1)
                if key.endswith('_p'):
                    y+=offset
                getattr(lib,key)[lo:hi]=y

        max_steps=int(command_steps.max()+max(lib.clip_len).item()+200)
        with torch.inference_mode():
            for tick in range(max_steps):
                if bool(done.all()):
                    break
                info=env.state.info['supermimic_state']
                before_s=info['s'].clone();before_a=info['a_prev'].clone()
                changed=[]
                active=(~done).cpu().numpy()
                for i,t in enumerate(trials):
                    if not active[i]:
                        continue
                    ci=t['candidate_index'];c=candidates[ci]
                    if stage[i]==0 and tick==command_steps[i]:
                        old=int(info['f'][i])
                        if old!=int(initial[i])+tick:
                            raise AssertionError('Source timeline changed before command')
                        if history is None:
                            raise AssertionError('Missing physical history')
                        if t['mode']=='direct_cut':
                            index=t['target_index'];anchor=int(lib.clip_start[index]);align(index,anchor,old)
                            stage[i]=2
                        else:
                            index=bridge_indices[ci]
                            anchor=int(lib.clip_start[index])+round((c['context_frames']-1)*50/30)
                            if ci not in aligned:
                                align(index,anchor,old);aligned.add(ci)
                            stage[i]=1
                        boundaries[i]=dict(reference_joint_jump_rad=float((lib.q[anchor]-lib.q[old]).square().mean().sqrt()),
                                            reference_height_jump_m=float(lib.root_p[anchor,2]-lib.root_p[old,2]))
                        env.f[i]=anchor;command_at[i]=tick;changed.append((i,'command'))
                        if t['mode'] in ['policy','zero_after_command']:
                            fi=ci+(C if t['mode']=='zero_after_command' else 0)
                            wm_initial[fi]=info['s'][i];wm_history[fi]=history[i];wm_prev[fi]=info['a_prev'][i]
                            wm_xi[fi]=info['xi'][i];wm_anchors[fi]=anchor;wm_available[fi]=True
                            actual_states[0,fi]=info['s'][i]
                    elif stage[i]==1:
                        bridge_index=bridge_indices[ci]
                        threshold=int(lib.clip_start[bridge_index])+round((c['context_frames']+c['bridge_frames'])*50/30)+50
                        if tick-command_at[i]>=127:
                            assert int(info['f'][i])==threshold
                            index=t['target_index'];anchor=int(lib.clip_start[index])+50
                            align(index,anchor,int(info['f'][i]));env.f[i]=anchor
                            stage[i]=2;handoff_at[i]=tick;changed.append((i,'canonical_handoff'))
                if changed:
                    old_reset_count=len(reset_ids)
                    old_history=history.clone()
                    env._publish();info=env.state.info['supermimic_state']
                    torch.testing.assert_close(info['s'],before_s,atol=1e-6,rtol=1e-6)
                    torch.testing.assert_close(info['a_prev'],before_a,atol=0,rtol=0)
                    torch.testing.assert_close(history,old_history,atol=0,rtol=0)
                    assert len(reset_ids)==old_reset_count
                    for i,kind in changed:
                        events.append(dict(trial=i,case=trials[i]['case'],mode=trials[i]['mode'],kind=kind,
                                           time_s=tick/50,physical_state_changed=False,action_changed=False,history_changed=False))
                frames=info['f'].clone()
                obs=mdp.obs(unflatten_state(info['s']),info['a_prev'],info['xi'],frames)
                history=obs[:,None].repeat(1,checkpoint['args']['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                action=policy(history,mdp.goal(frames,checkpoint['args']['goal_frames']))
                zero=torch.tensor([bool(t['mode']=='zero_after_command' and stage[i]>0) for i,t in enumerate(trials)],device=device)
                action=torch.where(zero[:,None],0.,action)
                reset_ids.clear()
                transition=env.step(action).info['supermimic']
                actual=unflatten_state(transition['s1'])
                ref_next=torch.minimum(frames+1,lib.clip_end[lib.clip_id[frames]])
                _,parts=mdp.reward(actual,ref_next,action,info['a_prev'],terms=True)
                done_np=transition['done'].cpu().numpy();term_np=transition['term'].cpu().numpy()
                speeds=actual[5].abs().amax(-1).cpu().numpy()
                reset_active={i for i in reset_ids if active[i]}
                expected_reset=set(np.flatnonzero(done_np&active).tolist())
                if reset_active!=expected_reset:
                    unexpected_resets.append(dict(tick=tick,actual=sorted(reset_active),expected=sorted(expected_reset)))
                    raise AssertionError('A physical reset occurred outside terminal lifecycle')
                for i in reset_active:
                    reset_counts[i]+=1
                reward=(transition['r']/mdp.max_step_reward).cpu().numpy()
                qsq=(actual[4]-lib.q[ref_next]).square().mean(-1).cpu().numpy()
                bsq=parts['e_body_pos'].cpu().numpy();rate=parts['action_rate'].cpu().numpy()
                root_sq=(actual[0]-lib.root_p[ref_next]).square().sum(-1).cpu().numpy()
                ori_sq=parts['e_body_ori'].cpu().numpy()
                forecast_trials=[i for i,t in enumerate(trials) if active[i] and command_at[i]>=0 and
                    tick-command_at[i]<H and t['mode'] in ['policy','zero_after_command']]
                if forecast_trials:
                    fis=torch.tensor([trials[i]['candidate_index']+(C if trials[i]['mode']=='zero_after_command' else 0) for i in forecast_trials],device=device)
                    offsets=torch.tensor([tick-command_at[i] for i in forecast_trials],device=device)
                    tis=torch.tensor(forecast_trials,device=device)
                    actual_states[offsets+1,fis]=transition['s1'][tis]
                    actual_actions[offsets,fis]=action[tis];actual_rewards[offsets,fis]=transition['r'][tis]/mdp.max_step_reward
                    actual_alive[offsets,fis]=True;actual_failed[offsets,fis]=transition['term'][tis]
                for i,t in enumerate(trials):
                    if not active[i]:
                        continue
                    peaks[i]=max(peaks[i],float(speeds[i]))
                    if command_at[i]>=0:
                        offset=tick-command_at[i]
                        transition_len=75 if t['mode']=='direct_cut' else 77
                        if offset<transition_len:
                            sums['transition_reward'][i]+=reward[i]
                            sums['transition_joint_sq'][i]+=qsq[i]
                            sums['transition_body_sq'][i]+=bsq[i]
                            sums['transition_root_sq'][i]+=root_sq[i];sums['transition_body_ori_sq'][i]+=ori_sq[i]
                            sums['transition_action_rate'][i]+=rate[i]
                            sums['transition_count'][i]+=1
                        if stage[i]==2:
                            sums['target_reward'][i]+=reward[i];sums['target_joint_sq'][i]+=qsq[i]
                            sums['target_body_sq'][i]+=bsq[i];sums['target_count'][i]+=1
                            sums['target_root_sq'][i]+=root_sq[i];sums['target_body_ori_sq'][i]+=ori_sq[i]
                    if bool(done_np[i]):
                        failed[i]=bool(term_np[i]);final_at[i]=tick
                        completed[i]=not failed[i] and stage[i]==2
                        if completed[i]:
                            assert int(ref_next[i])==int(lib.clip_end[t['target_index']])
                if video_indices:
                    mask=torch.tensor([bool(active[i]) for i in video_indices],device=device)
                    video_pose.append(torch.where(mask[:,None],flattened_qpos(transition['s1'])[video_indices],float('nan')).cpu().numpy())
                    video_ref.append(torch.where(mask[:,None],reference_qpos(lib,ref_next)[video_indices],float('nan')).cpu().numpy())
                    video_alive.append(mask.cpu().numpy());video_stage.append(stage[video_indices].copy())
                done|=transition['done']
                if tick%500==0:
                    print('live tick',tick,'completed',int(completed.sum()),'failed',int(failed.sum()),'of',trial_count,flush=True)
        rows=[]
        for i,t in enumerate(trials):
            denom=75 if t['mode']=='direct_cut' else 77
            target_denom=max(1,int(lib.clip_len[t['target_index']])-1-(50 if t['mode']!='direct_cut' else 0))
            transition_score=sums['transition_reward'][i]/denom
            target_score=sums['target_reward'][i]/target_denom
            rows.append(dict(**{k:v for k,v in t.items() if k not in ['candidate_index','target_index']},
                completed=bool(completed[i]),failed=bool(failed[i]),command_issued=bool(command_at[i]>=0),
                command_time_s=int(command_at[i])/50,handoff_time_s=int(handoff_at[i])/50,
                final_time_s=int(final_at[i])/50,transition_score=float(transition_score),target_score=float(target_score),
                quality_pass=bool(completed[i] and transition_score>=.8 and target_score>=.8),
                transition_joint_rmse=float((sums['transition_joint_sq'][i]/max(1,sums['transition_count'][i]))**.5),
                transition_relative_body_rmse_m=float((sums['transition_body_sq'][i]/max(1,sums['transition_count'][i]))**.5),
                transition_root_rmse_m=float((sums['transition_root_sq'][i]/max(1,sums['transition_count'][i]))**.5),
                transition_body_orientation_rmse_rad=float((sums['transition_body_ori_sq'][i]/max(1,sums['transition_count'][i]))**.5),
                mean_action_rate_sq=float(sums['transition_action_rate'][i]/max(1,sums['transition_count'][i])),
                target_joint_rmse=float((sums['target_joint_sq'][i]/max(1,sums['target_count'][i]))**.5),
                target_relative_body_rmse_m=float((sums['target_body_sq'][i]/max(1,sums['target_count'][i]))**.5),
                target_root_rmse_m=float((sums['target_root_sq'][i]/max(1,sums['target_count'][i]))**.5),
                target_body_orientation_rmse_rad=float((sums['target_body_ori_sq'][i]/max(1,sums['target_count'][i]))**.5),
                peak_joint_speed_rad_s=float(peaks[i]),terminal_reset_count=int(reset_counts[i]),**boundaries[i]))
        (a.output_dir/'live_trials.json').write_text(json.dumps(dict(rows=rows,events=events,unexpected_resets=unexpected_resets),indent=2,allow_nan=False,default=json_scalar)+'\n')
        np.savez_compressed(a.output_dir/'videos.npz',policy=np.asarray(video_pose),reference=np.asarray(video_ref),alive=np.asarray(video_alive),stage=np.asarray(video_stage),trial_indices=np.asarray(video_indices),fps=50)
        torch.save(dict(initial=wm_initial,history=wm_history,previous_action=wm_prev,context=wm_xi,anchors=wm_anchors,
                        actual_states=actual_states,actual_actions=actual_actions,actual_rewards=actual_rewards,
                        actual_alive=actual_alive,actual_failed=actual_failed,available=torch.tensor(wm_available)),a.output_dir/'wm_audit_inputs.pt')
        # Independent free WM rollout from the actual, uninterrupted command state.
        wm=networks.WorldModel(hidden=checkpoint['args']['wm_hidden'],num_joints=mdp.num_joints,context_dim=mdp.context_dim).to(device)
        wm.load_state_dict(checkpoint['wm']);wm.eval()
        valid=[i for i in range(F) if wm_available[i]]
        forecasts=[]
        if valid:
            ids=torch.tensor(valid,device=device);state=unflatten_state(wm_initial[ids])
            hist=wm_history[ids].clone();prev=wm_prev[ids].clone();xi=wm_xi[ids];frames=wm_anchors[ids].clone()
            held_state=tuple(x.clone() for x in state);held_history=hist.clone();held_prev=prev.clone()
            zero_mode=ids>=C
            live=torch.ones(len(valid),dtype=torch.bool,device=device)
            predicted_reward=[];predicted_failed=[];predicted_qerr=[];held_rewards=[]
            with torch.inference_mode():
                for step in range(H):
                    obs=mdp.obs(state,prev,xi,frames);hist=torch.cat((hist[:,1:],obs[:,None]),1)
                    action=policy(hist,mdp.goal(frames,checkpoint['args']['goal_frames']))
                    action=torch.where(zero_mode[:,None],0.,action)
                    next_state=wm.step(state,action,mdp.normalize_context(xi))
                    reward=mdp.reward(next_state,frames+1,action,prev)/mdp.max_step_reward
                    term=mdp.termination(next_state,frames+1)
                    predicted_reward.append(torch.where(live,reward,0.).cpu().numpy())
                    predicted_failed.append((live&term).cpu().numpy())
                    actual=unflatten_state(actual_states[step+1,ids])
                    predicted_qerr.append((next_state[4]-actual[4]).square().mean(-1).cpu().numpy())
                    held_obs=mdp.obs(held_state,held_prev,xi,frames)
                    held_history=torch.cat((held_history[:,1:],held_obs[:,None]),1)
                    held_action=policy(held_history,mdp.goal(frames,checkpoint['args']['goal_frames']))
                    held_action=torch.where(zero_mode[:,None],0.,held_action)
                    held_rewards.append((mdp.reward(held_state,frames+1,held_action,held_prev)/mdp.max_step_reward).cpu().numpy())
                    held_prev=held_action
                    live&=~term;state=next_state;prev=action;frames+=1
            pr=np.stack(predicted_reward);pf=np.stack(predicted_failed);pe=np.stack(predicted_qerr);hr=np.stack(held_rewards)
            for j,fi in enumerate(valid):
                ci=fi%C
                for kind,lo,hi in [('prefix32',0,32),('bridge',1,76),('target_1s',76,126)]:
                    mask=actual_alive[lo:hi,fi].cpu().numpy()
                    forecasts.append(dict(case=candidates[ci]['name'],kind=kind,mode='zero_after_command' if fi>=C else 'policy',
                        wm_return=float(pr[lo:hi,j].mean()),sim_return=float(actual_rewards[lo:hi,fi].mean()),
                        frozen_state_return=float(hr[lo:hi,j].mean()),
                        wm_failed=bool(pf[:hi,j].any()),sim_failed=bool(actual_failed[:hi,fi].any()),
                        free_prediction_joint_rmse=float(np.sqrt(pe[lo:hi,j][mask].mean())) if mask.any() else None))
        summary={}
        for mode in a.modes:
            rs=[r for r in rows if r['mode']==mode]
            summary[mode]=dict(n=len(rs),completed=sum(r['completed'] for r in rs),failed=sum(r['failed'] for r in rs),
                quality_pass=sum(r['quality_pass'] for r in rs),mean_transition_score=float(np.mean([r['transition_score'] for r in rs])),
                mean_target_score=float(np.mean([r['target_score'] for r in rs])),
                mean_transition_joint_rmse=float(np.mean([r['transition_joint_rmse'] for r in rs])),
                mean_transition_body_rmse_m=float(np.mean([r['transition_relative_body_rmse_m'] for r in rs])))
        result=dict(protocol='A starts at frame zero; single uninterrupted physics run; no cache restoration; reference-only command and B handoff; no training',
            checkpoint=str(checkpoint_path.resolve()),checkpoint_sha256=digest,iteration=checkpoint['iteration'],
            phase_seconds=sorted({c['phase_s'] for c in candidates}),summary=summary,
            unexpected_resets=unexpected_resets,events=events,rows=rows,forecasts=forecasts,
            wm_summary={kind:correlations([r for r in forecasts if r['kind']==kind and r['mode']=='policy']) for kind in ['prefix32','bridge','target_1s']},
            wm_negative_summary={kind:correlations([r for r in forecasts if r['kind']==kind and r['mode']=='zero_after_command']) for kind in ['prefix32','bridge','target_1s']},
            quality_definition='Exploratory quality screen: hard completion AND transition window reward>=0.8 AND remaining target reward>=0.8; not a deployment certificate')
        (a.output_dir/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False,default=json_scalar)+'\n')
        with (a.output_dir/'trials.csv').open('w') as stream:
            fields=sorted({k for r in rows for k in r});w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
