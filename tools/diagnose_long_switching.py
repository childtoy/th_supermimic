#!/usr/bin/env python3
"""Frozen long-base switching diagnosis: cache actual source states, build bridges,
then compare real physics to closed-loop WM forecasts. No training occurs.
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


def prepare(a):
    import numpy as np
    import torch
    from th_supermimic.sim.env import TrackingEnv
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import load_policy
    from th_supermimic.rl.algos.supermimic.geometry import unflatten_state
    if a.output_dir.exists():
        raise FileExistsError('Use a new output directory')
    a.output_dir.mkdir(parents=True)
    checkpoint=torch.load(a.base_dir/'train/best_coverage.pt',map_location='cpu',weights_only=True)
    manifest=json.loads((a.base_dir/'data/manifest.json').read_text())
    records=manifest['clips']; groups=[r['name'].split('__')[0] for r in records]
    files=[str(a.base_dir/'data'/(r['name']+'.csv')) for r in records]
    torch.manual_seed(0); torch.set_num_threads(1)
    env=TrackingEnv(a.assets_dir,files,len(files),backend='mjwarp',eval_mode=True,randomization=False)
    caches=[]
    try:
        policy=load_policy(env,checkpoint);mdp=env.supermimic_mdp
        env.reset_to_frames(env.motion.clip_start.clone());history=None
        with torch.inference_mode():
            for step in range(round(max(a.phases)*50)+1):
                info=env.state.info['supermimic_state']
                if step in {round(p*50) for p in a.phases}:
                    if history is None:
                        raise ValueError('Use positive source phases')
                    for i,group in enumerate(groups):
                        frame=int(info['f'][i])
                        expected=int(env.motion.clip_start[i])+step
                        if frame!=expected:
                            raise RuntimeError(f'{group} failed before requested source phase')
                        caches.append(dict(group=group,phase_s=step/50,s=info['s'][i].cpu(),
                            history_before=history[i].cpu(),a_prev=info['a_prev'][i].cpu(),xi=info['xi'][i].cpu(),
                            ref_p=env.motion.root_p[frame].cpu(),ref_R=env.motion.root_R[frame].cpu(),
                            ref_q=env.motion.q[frame].cpu(),ref_qd=env.motion.qd[frame].cpu()))
                if step==round(max(a.phases)*50):
                    break
                obs=mdp.obs(unflatten_state(info['s']),info['a_prev'],info['xi'],info['f'])
                history=obs[:,None].repeat(1,checkpoint['args']['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                result=env.step(policy(history,mdp.goal(info['f'],checkpoint['args']['goal_frames']))).info['supermimic']
                if bool(result['done'].any()):
                    raise RuntimeError('Source tracking ended before all requested phases')
        target_q={group:env.motion.q[env.motion.clip_start[i]].cpu() for i,group in enumerate(groups)}
    finally:
        env.close()
    torch.save(dict(caches=caches,target_q=target_q,training_contract=checkpoint['contract'],
                    checkpoint_iteration=checkpoint['iteration']),a.output_dir/'source_cache.pt')
    print('Cached actual source states/history:',len(caches),flush=True)
    sys.path.insert(0,str(a.rmai_root.resolve()))
    sys.path.insert(0,str(a.corpus_root.resolve()/'scripts'))
    from motion_geometry import normalize_root
    from scripts.run_three_demos_soma_transition_comparison import LearnedPlacement,load_inbetweener,generate_transition,transform_sequence_at_anchor
    from rmai.core.kimodo_features import features_to_kimodo_npz
    long_manifest=json.loads((a.base_dir/'reference_build/manifest.json').read_text())
    humans={r['group']:np.load(a.base_dir/'reference_build/features'/r['group']/(r['name']+'.npz'))['features'] for r in long_manifest['clips']}
    settings=json.loads((a.corpus_root/'configs/alignment.json').read_text())
    learned=LearnedPlacement(a.rmai_root/'output/alignment_rank_v3_soma30_v7_xl/best.pt',torch.device('cuda'),30)
    inb=load_inbetweener(a.rmai_root/'output/inbetweening_soma30_bs512/latest.pt',a.rmai_root/'data/processed/soma30_fc_parallel/normalizer.npz',torch.device('cuda'))
    context=inb[3].context_frames;bridge=45
    if bridge not in inb[3].bridge_lengths or context!=learned.K:
        raise ValueError('RMAI bridge/context mismatch')
    cases=[]
    with torch.inference_mode():
        for cache_index,cache in enumerate(caches):
            source=cache['group'];stop=round(cache['phase_s']*30)+1
            left=humans[source][stop-context:stop]
            for target in groups:
                if target==source:
                    continue
                torch.manual_seed(42+len(cases));np.random.seed(42+len(cases))
                right=humans[target][:61]
                placement=learned.choose(left,right[:context],[1.5],settings['alignment_policy'])
                placed=transform_sequence_at_anchor(right,0,np.asarray(placement['target_planar'],np.float32),float(placement['target_heading_rad']),30)
                generated=generate_transition(left,placed[:context],bridge,*inb,torch.device('cuda'),20)
                full=np.concatenate((left,generated[context:context+bridge],placed),axis=0)
                full,normalization=normalize_root(full)
                if not np.isfinite(full).all():
                    raise ValueError('Invalid generated bridge')
                name=f'{source}__to__{target}__t{round(cache["phase_s"]*100):05d}'
                native=a.output_dir/'soma77'/source/(name+'.npz')
                features_to_kimodo_npz(full,native,num_joints=30,fps=30,kimodo_root=a.kimodo_root,export_num_joints=77)
                cases.append(dict(name=name,source=source,target=target,phase_s=cache['phase_s'],cache_index=cache_index,
                    context_frames=context,bridge_frames=bridge,source_frames=len(full),fps=30,
                    native=str(native.resolve()),placement=placement,normalization=normalization))
                (a.output_dir/'cases.json').write_text(json.dumps(dict(cases=cases,
                    checkpoint=str((a.base_dir/'train/best_coverage.pt').resolve()),
                    checkpoint_sha256=hashlib.sha256((a.base_dir/'train/best_coverage.pt').read_bytes()).hexdigest(),
                    protocol='Fresh RMAI bridge, local independent retarget; actual cached source state/history retained; no training'),indent=2)+'\n')
                print('Generated',name,flush=True)


def evaluate(a):
    import numpy as np
    import torch
    from th_supermimic.data.rmai import convert_file
    from th_supermimic.sim.env import TrackingEnv
    from th_supermimic.data.playback import for_new_references
    from th_supermimic.rl.algos.supermimic import networks
    from th_supermimic.rl.algos.supermimic.geometry import unflatten_state,mat_to_quat,yaw_mat,flatten_state
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import load_policy
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import flattened_qpos,reference_qpos,render_comparison
    from calibrate_transition_wm import correlations
    torch.set_num_threads(1)
    bundle=json.loads((a.output_dir/'cases.json').read_text());cases=list(bundle['cases'])
    report_stem='result_full_follow' if a.continue_full_target else 'result'
    if (a.output_dir/(report_stem+'.json')).exists():
        raise FileExistsError('Evaluation already exists')
    checkpoint=torch.load(a.base_dir/'train/best_coverage.pt',map_location='cpu',weights_only=True)
    if hashlib.sha256((a.base_dir/'train/best_coverage.pt').read_bytes()).hexdigest()!=bundle['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after source caching')
    saved=torch.load(a.output_dir/'source_cache.pt',map_location='cpu',weights_only=True)
    base_records=json.loads((a.base_dir/'data/manifest.json').read_text())['clips']
    base_sources={r['name'].split('__')[0]:a.base_dir/'robots_mj360'/r['name'].split('__')[0]/('long_'+r['name'].split('__')[0])/'k1/final/robot_motion.npz' for r in base_records}
    for i,cache in enumerate(saved['caches']):
        cases.append(dict(name=f'control__{cache["group"]}__t{round(cache["phase_s"]*100):05d}',
            source=cache['group'],target=cache['group'],phase_s=cache['phase_s'],cache_index=i,control=True))
    files=[]
    for case in cases:
        path=base_sources[case['source']] if case.get('control') else a.output_dir/'robots_mj360'/case['source']/case['name']/'k1/final/robot_motion.npz'
        values,_=convert_file(path,'k1')
        csv_path=a.output_dir/'csv'/(case['name']+'.csv');csv_path.parent.mkdir(exist_ok=True)
        np.savetxt(csv_path,values,delimiter=',',fmt='%.10g');files.append(str(csv_path))
    follow_indices={}
    if a.continue_full_target:
        for i,case in enumerate(cases):
            if case.get('control'):
                continue
            values,_=convert_file(base_sources[case['target']],'k1')
            path=a.output_dir/'csv'/('follow__'+case['name']+'.csv')
            np.savetxt(path,values,delimiter=',',fmt='%.10g')
            follow_indices[i]=len(files);files.append(str(path))
    env=TrackingEnv(a.assets_dir,files,len(cases),backend='mjwarp',eval_mode=True,randomization=False)
    try:
        policy=load_policy(env,for_new_references(checkpoint,env.contract));lib=env.motion;mdp=env.supermimic_mdp
        wm=networks.WorldModel(hidden=checkpoint['args']['wm_hidden'],num_joints=mdp.num_joints,context_dim=mdp.context_dim).to(env.device)
        wm.load_state_dict(checkpoint['wm']);wm.eval()
        n=len(cases); device=env.device
        cache=[saved['caches'][case['cache_index']] for case in cases]
        state=unflatten_state(torch.stack([c['s'] for c in cache]).to(device))
        history=torch.stack([c['history_before'] for c in cache]).to(device)
        prev=torch.stack([c['a_prev'] for c in cache]).to(device)
        xi=torch.stack([c['xi'] for c in cache]).to(device)
        anchors=[]; bridge_start=[]; bridge_end=[]; boundaries=[]
        with torch.inference_mode():
            canonical_target_1s={}
            for i,case in enumerate(cases):
                if case.get('control'):
                    canonical_target_1s[case['source']]=lib.q[lib.clip_start[i]+50].clone()
            for i,case in enumerate(cases):
                lo=int(lib.clip_start[i]);hi=int(lib.clip_end[i])+1
                if case.get('control'):
                    anchor=lo+round(case['phase_s']*50);first=anchor+1;end=first+75
                else:
                    anchor=lo+round((case['context_frames']-1)*50/30)
                    first=lo+round(case['context_frames']*50/30)
                    end=lo+round((case['context_frames']+case['bridge_frames'])*50/30)
                rotation=yaw_mat(cache[i]['ref_R'].to(device))@yaw_mat(lib.root_R[anchor]).T
                offset=cache[i]['ref_p'].to(device)-rotation@lib.root_p[anchor];offset[2]=0
                for key in ['root_p','root_R','root_v','root_w','body_p','body_R','body_v','body_w']:
                    value=getattr(lib,key)[lo:hi].clone()
                    moved=rotation@value if key.endswith('_R') else (rotation@value[...,None]).squeeze(-1)
                    if key.endswith('_p'):
                        moved+=offset
                    getattr(lib,key)[lo:hi]=moved
                boundaries.append(dict(source_reference_joint_jump_rad=float((lib.q[anchor]-cache[i]['ref_q'].to(device)).square().mean().sqrt()),
                    source_reference_height_jump_m=float(lib.root_p[anchor,2]-cache[i]['ref_p'][2].to(device)),
                    actual_initial_joint_error_rad=float((lib.q[anchor]-state[4][i]).square().mean().sqrt()),
                    source_velocity_jump_rad_s=float((lib.qd[anchor]-cache[i]['ref_qd'].to(device)).square().mean().sqrt()),
                    target_canonical_joint_gap_rad=None if case.get('control') else float((lib.q[end]-saved['target_q'][case['target']].to(device)).square().mean().sqrt()),
                    target_canonical_joint_gap_1s_rad=None if case.get('control') else float((lib.q[end+50]-canonical_target_1s[case['target']]).square().mean().sqrt())))
                if case.get('control') and boundaries[-1]['source_reference_joint_jump_rad']>1e-4:
                    raise AssertionError('Known-reference control changed its source pose')
                anchors.append(anchor);bridge_start.append(first-anchor-1);bridge_end.append(end-anchor-1)
            ids=torch.arange(n,device=device)
            anchors=torch.tensor(anchors,device=device)
            horizon=max(bridge_end)+51
            qpos=torch.tensor(env.sim.model.qpos0,device=device,dtype=torch.float32)[None].repeat(n,1)
            qvel=torch.zeros(n,env.sim.model.nv,device=device)
            qpos[:,:3]=state[0];qpos[:,3:7]=mat_to_quat(state[1]);qpos[:,env.sim.qadr]=state[4]
            qvel[:,:3]=state[2];qvel[:,3:6]=(state[1].transpose(-1,-2)@state[3][...,None]).squeeze(-1);qvel[:,env.sim.vadr]=state[5]
            env.xi.copy_(xi);env.sim.randomize(ids,xi[:,0],xi[:,1+mdp.num_joints:]);env.sim.reset(ids,qpos,qvel)
            env.f.copy_(anchors);env.a_prev.copy_(prev);env.ep_step.zero_();env.ep_ret.zero_();env.ep_disc.zero_();env._publish()
            restored=unflatten_state(env.state.info['supermimic_state']['s'])
            for original,actual in zip(state,restored):
                torch.testing.assert_close(actual,original,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(env.a_prev,prev,atol=0,rtol=0)
            predicted_state=tuple(x.clone() for x in state)
            wm_prev=prev.clone();wm_hist=history.clone();sim_hist=history.clone()
            norm_xi=mdp.normalize_context(xi)
            sim_alive=torch.ones(n,dtype=torch.bool,device=device);wm_alive=sim_alive.clone()
            traces={key:[] for key in ['sim_reward','wm_reward','sim_failed','wm_failed','sim_alive','wm_alive',
                                       'sim_joint_sq','wm_joint_sq','prediction_joint_sq','one_step_joint_sq','prediction_root_sq','ref','sim','wm']}
            for step in range(horizon):
                frames=anchors+step;future=frames+1
                current=env.state.info['supermimic_state'];real=unflatten_state(current['s'])
                obs=mdp.obs(real,current['a_prev'],xi,frames)
                sim_hist=torch.cat((sim_hist[:,1:],obs[:,None]),1)
                action=policy(sim_hist,mdp.goal(frames,checkpoint['args']['goal_frames']))
                obs_wm=mdp.obs(predicted_state,wm_prev,xi,frames)
                wm_hist=torch.cat((wm_hist[:,1:],obs_wm[:,None]),1)
                wm_action=policy(wm_hist,mdp.goal(frames,checkpoint['args']['goal_frames']))
                prediction=wm.step(predicted_state,wm_action,norm_xi)
                one_step=wm.step(real,action,norm_xi)
                predicted_reward=mdp.reward(prediction,future,wm_action,wm_prev)
                predicted_failure=mdp.termination(prediction,future)
                finite=torch.stack([x.isfinite().flatten(1).all(1) for x in prediction]).all(0)
                predicted_failure|=~finite
                result=env.step(action).info['supermimic'];actual=unflatten_state(result['s1'])
                values=dict(sim_reward=torch.where(sim_alive,result['r']/mdp.max_step_reward,0.),
                    wm_reward=torch.where(wm_alive,torch.nan_to_num(predicted_reward/mdp.max_step_reward,nan=0.,posinf=0.,neginf=0.),0.),
                    sim_failed=sim_alive&result['term'],wm_failed=wm_alive&predicted_failure,
                    sim_alive=sim_alive.clone(),wm_alive=wm_alive.clone(),
                    sim_joint_sq=(actual[4]-lib.q[future]).square().mean(-1),
                    wm_joint_sq=(prediction[4]-lib.q[future]).square().mean(-1),
                    prediction_joint_sq=(prediction[4]-actual[4]).square().mean(-1),
                    one_step_joint_sq=(one_step[4]-actual[4]).square().mean(-1),
                    prediction_root_sq=(prediction[0]-actual[0]).square().sum(-1),
                    ref=reference_qpos(lib,future),sim=flattened_qpos(result['s1']),wm=flattened_qpos(flatten_state(prediction)))
                for key,value in values.items():
                    traces[key].append(value.cpu().numpy())
                sim_alive&=~result['done'];wm_alive&=~predicted_failure
                predicted_state=prediction;wm_prev=wm_action
            traces={key:np.stack(value) for key,value in traces.items()}
        rows=[]
        long_segments={r['group']:r['segments'] for r in json.loads((a.base_dir/'reference_build/manifest.json').read_text())['clips']}
        for i,case in enumerate(cases):
            source_frame=round(case['phase_s']*30)
            segment=next(s for s in long_segments[case['source']] if s['start']<=source_frame<s['end'])
            windows={'prefix32':(0,32),'bridge':(bridge_start[i],bridge_end[i]),
                     'target_1s':(bridge_end[i],bridge_end[i]+50)}
            for kind,(lo,hi) in windows.items():
                sa=traces['sim_alive'][lo:hi,i];wa=traces['wm_alive'][lo:hi,i];common=sa&wa
                def rms(key,mask):
                    values=traces[key][lo:hi,i][mask]
                    return float(np.sqrt(values.mean())) if len(values) and np.isfinite(values).all() else None
                rows.append(dict(case=case['name'],source=case['source'],target=case['target'],phase_s=case['phase_s'],kind=kind,
                    control=case.get('control',False),
                    source_segment_kind=segment['kind'],
                    sim_return=float(traces['sim_reward'][lo:hi,i].mean()),wm_return=float(traces['wm_reward'][lo:hi,i].mean()),
                    sim_failed=bool(traces['sim_failed'][:hi,i].any()),wm_failed=bool(traces['wm_failed'][:hi,i].any()),
                    sim_observed_steps=int(sa.sum()),horizon_steps=hi-lo,
                    sim_joint_rmse=rms('sim_joint_sq',sa),wm_joint_rmse=rms('wm_joint_sq',wa),
                    free_prediction_joint_rmse=rms('prediction_joint_sq',common),one_step_prediction_joint_rmse=rms('one_step_joint_sq',sa),
                    free_prediction_root_rmse_m=rms('prediction_root_sq',common),**boundaries[i]))
        summary={kind:correlations([r for r in rows if r['kind']==kind and not r['control']]) for kind in ['prefix32','bridge','target_1s']}
        summary['known_continuation']={kind:correlations([r for r in rows if r['kind']==kind and r['control']]) for kind in ['prefix32','bridge','target_1s']}
        cross_ids=[i for i,c in enumerate(cases) if not c.get('control')]
        summary['cases']=len(cross_ids);summary['controls']=n-len(cross_ids)
        summary['sim_failures_by_end']=int(traces['sim_failed'][:,cross_ids].any(0).sum())
        summary['wm_failures_by_end']=int(traces['wm_failed'][:,cross_ids].any(0).sum())
        result=dict(protocol=bundle['protocol'],checkpoint=bundle['checkpoint'],iteration=checkpoint['iteration'],
                    source_cache='Actual base-policy execution at each source phase; state, a_prev and observation history retained',
                    summary=summary,rows=rows,cases=cases)
        if a.continue_full_target:
            with torch.inference_mode():
                current=env.state.info['supermimic_state']
                failed_before=~sim_alive.clone()
                handoffs={}
                for i,index in follow_indices.items():
                    if not bool(sim_alive[i]):
                        continue
                    old_frame=int(current['f'][i]);lo=int(lib.clip_start[index]);hi=int(lib.clip_end[index])+1
                    new_frame=lo+50
                    rotation=yaw_mat(lib.root_R[old_frame])@yaw_mat(lib.root_R[new_frame]).T
                    offset=lib.root_p[old_frame]-rotation@lib.root_p[new_frame];offset[2]=0
                    for key in ['root_p','root_R','root_v','root_w','body_p','body_R','body_v','body_w']:
                        value=getattr(lib,key)[lo:hi].clone()
                        moved=rotation@value if key.endswith('_R') else (rotation@value[...,None]).squeeze(-1)
                        if key.endswith('_p'):
                            moved+=offset
                        getattr(lib,key)[lo:hi]=moved
                    handoffs[i]=dict(joint_jump_rad=float((lib.q[old_frame]-lib.q[new_frame]).square().mean().sqrt()),
                        height_jump_m=float(lib.root_p[new_frame,2]-lib.root_p[old_frame,2]))
                    env.f[i]=new_frame
                # Change only the reference index; physical state/action/history stay live.
                env._publish()
                start_frames=env.f.clone();ends=lib.clip_end[lib.clip_id[start_frames]]
                expected=(ends-start_frames).clamp(min=1)
                follow_return=torch.zeros(n,device=device);follow_steps=torch.zeros(n,dtype=torch.long,device=device)
                follow_failed=failed_before.clone();completed=torch.zeros(n,dtype=torch.bool,device=device)
                follow_poses=[];follow_refs=[];follow_masks=[]
                for _ in range(int(expected.max())+1):
                    if not bool(sim_alive.any()):
                        break
                    current=env.state.info['supermimic_state']
                    obs=mdp.obs(unflatten_state(current['s']),current['a_prev'],xi,current['f'])
                    sim_hist=torch.cat((sim_hist[:,1:],obs[:,None]),1)
                    action=policy(sim_hist,mdp.goal(current['f'],checkpoint['args']['goal_frames']))
                    reference_frames=torch.minimum(current['f']+1,lib.clip_end[lib.clip_id[current['f']]])
                    transition=env.step(action).info['supermimic']
                    follow_return+=torch.where(sim_alive,transition['r'],0.)
                    follow_steps+=sim_alive
                    follow_failed|=sim_alive&transition['term']
                    completed|=sim_alive&transition['done']&~transition['term']
                    follow_poses.append(flattened_qpos(transition['s1']).cpu().numpy())
                    follow_refs.append(reference_qpos(lib,reference_frames).cpu().numpy())
                    follow_masks.append(sim_alive.cpu().numpy().copy())
                    sim_alive&=~transition['done']
                follow_rows=[]
                for i,case in enumerate(cases):
                    follow_rows.append(dict(case=case['name'],control=case.get('control',False),
                        target=case['target'],completed=bool(completed[i]),failed=bool(follow_failed[i]),
                        failed_before_handoff=bool(failed_before[i]),steps=int(follow_steps[i]),expected_steps=int(expected[i]),
                        norm_return=float(follow_return[i]/expected[i]/mdp.max_step_reward),**handoffs.get(i,{})))
                cross_follow=[r for r in follow_rows if not r['control']]
                result['full_target_follow']=dict(completed=sum(r['completed'] for r in cross_follow),
                    failed=sum(r['failed'] for r in cross_follow),cases=len(cross_follow),
                    mean_remaining_target_return=sum(r['norm_return'] for r in cross_follow)/len(cross_follow),
                    protocol='After local B first second, hand off to a unique copy of the original trained full B at 1s; planar reference alignment only; no physical/action/history reset',
                    rows=follow_rows)
                np.savez_compressed(a.output_dir/'full_follow_traces.npz',policy=np.asarray(follow_poses),reference=np.asarray(follow_refs),alive=np.asarray(follow_masks),fps=50)
                print('FULL_TARGET_FOLLOW',json.dumps({k:v for k,v in result['full_target_follow'].items() if k!='rows'}),flush=True)
        (a.output_dir/(report_stem+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        with (a.output_dir/(report_stem+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        np.savez_compressed(a.output_dir/(report_stem+'_traces.npz' if a.continue_full_target else 'traces.npz'),**traces)
        print(json.dumps(summary,indent=2),flush=True)
        if a.video:
            bridge_rows=[r for r in rows if r['kind']=='bridge']
            chosen={min(cross_ids,key=lambda i:bridge_rows[i]['sim_return']),max(cross_ids,key=lambda i:bridge_rows[i]['sim_return']),
                    max(cross_ids,key=lambda i:abs(bridge_rows[i]['sim_return']-bridge_rows[i]['wm_return']))}
            for i in sorted(chosen):
                observed=np.flatnonzero(traces['sim_alive'][:,i]);end=int(observed[-1])+1 if len(observed) else 1
                render_comparison(traces['ref'][:end,i],traces['sim'][:end,i],a.output_dir/'videos'/(cases[i]['name']+'.mp4'),
                    a.assets_dir/'scene.xml',clip_name=cases[i]['name'],width=480,height=288,fps=10,global_ghost=True)
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','evaluate'])
    p.add_argument('--base-dir',type=Path,default=Path('runs/long_full_base_20261103'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--rmai-root',type=Path,default=Path('../RMAI'))
    p.add_argument('--corpus-root',type=Path,default=Path('/workspace/primitive-motion-corpus'))
    p.add_argument('--kimodo-root',type=Path,default=Path('../kimodo'))
    p.add_argument('--assets-dir',type=Path,default=Path('assets/k1'))
    p.add_argument('--phases',type=float,nargs='+',default=[5,15,25])
    p.add_argument('--video',action='store_true')
    p.add_argument('--continue-full-target',action='store_true')
    a=p.parse_args()
    os.environ.setdefault('MUJOCO_GL','egl')
    (prepare if a.stage=='prepare' else evaluate)(a)


if __name__=='__main__':
    main()
