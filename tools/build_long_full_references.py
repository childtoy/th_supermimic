#!/usr/bin/env python3
"""Compose three >=30s human references using the existing RMAI models.

These are new sequences of existing short primitives and generated bridges,
not recovered historical full files or independently captured long motions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


ORDERS = {
    'boxing': ['straight_punch', 'double_uppercut', 'duck', 'weave'],
    'traffic_control1': ['stop_a', 'come_forward', 'direct_left', 'turn_right_90', 'turn_around'],
    'traffic_control2': ['firm_stop', 'come_forward', 'direct_left', 'slow_down'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rmai-root', type=Path, required=True)
    p.add_argument('--corpus-root', type=Path, required=True)
    p.add_argument('--kimodo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--min-seconds', type=float, default=30.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--inference-steps', type=int, default=20)
    a = p.parse_args()
    if a.output_dir.exists():
        p.error('Use a new output directory')
    if a.min_seconds<1:
        p.error('min-seconds must be positive')
    import numpy as np
    import torch
    sys.path.insert(0, str(a.rmai_root.resolve()))
    sys.path.insert(0, str(a.corpus_root.resolve()/'scripts'))
    from motion_geometry import normalize_root
    from scripts.run_three_demos_soma_transition_comparison import (
        LearnedPlacement, load_inbetweener, generate_transition, transform_sequence_at_anchor,
    )
    from rmai.core.kimodo_features import features_to_kimodo_npz
    alignment = a.rmai_root/'output/alignment_rank_v3_soma30_v7_xl/best.pt'
    inbetweening = a.rmai_root/'output/inbetweening_soma30_bs512/latest.pt'
    normalizer = a.rmai_root/'data/processed/soma30_fc_parallel/normalizer.npz'
    settings = json.loads((a.corpus_root/'configs/alignment.json').read_text())
    device = torch.device('cuda')
    learned = LearnedPlacement(alignment,device,30)
    inb = load_inbetweener(inbetweening,normalizer,device)
    config = inb[3]; context=config.context_frames
    if context!=learned.K or config.representation!='kimodo_soma30':
        raise ValueError('RMAI checkpoint context/representation mismatch')
    durations=[d for d in settings['duration_candidates_seconds'] if round(d*30) in config.bridge_lengths]
    if not durations:
        raise ValueError('No supported bridge duration')
    sources={}
    for group in ORDERS:
        entries=json.loads((a.corpus_root/'configs'/(group+'.json')).read_text())['primitives']
        clips={}
        for entry in entries:
            stored=entry['source_file']
            path=a.rmai_root/('data/'+stored.split('/data/',1)[1] if '/data/' in stored else stored)
            with np.load(path,allow_pickle=False) as archive:
                values=np.asarray(archive['features'],np.float32)
                fps=float(archive['fps'].reshape(-1)[0]) if 'fps' in archive else 30.
            if fps!=30 or values.ndim!=2 or values.shape[1]!=369 or len(values)<context or not np.isfinite(values).all():
                raise ValueError(f'Invalid source {path}')
            clips[entry['action']]=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),features=values)
        sources[group]=clips
    a.output_dir.mkdir(parents=True)
    signature=dict(seed=a.seed,min_seconds=a.min_seconds,inference_steps=a.inference_steps,
                   weights={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [alignment,inbetweening,normalizer]})
    manifest=dict(protocol='Existing short primitives repeated in a fixed order with RMAI-generated bridges; no cross-scenario transitions',settings=signature,clips=[])
    with torch.inference_mode():
        for group,order in ORDERS.items():
            torch.manual_seed(a.seed);np.random.seed(a.seed)
            clips=sources[group]; full=clips[order[0]]['features'].copy()
            segments=[dict(kind='primitive',action=order[0],start=0,end=len(full))]
            edges=[]; next_index=1
            while (len(full)-1)/30<a.min_seconds:
                target=order[next_index%len(order)]
                values=clips[target]['features']
                placement=learned.choose(full[-context:],values[:context],durations,settings['alignment_policy'])
                placed=transform_sequence_at_anchor(values,0,np.asarray(placement['target_planar'],np.float32),float(placement['target_heading_rad']),30)
                bridge_frames=int(placement['bridge_frames'])
                generated=generate_transition(full[-context:],placed[:context],bridge_frames,*inb,device,a.inference_steps)
                bridge=np.asarray(generated[context:context+bridge_frames],np.float32)
                if len(bridge)!=bridge_frames or not np.isfinite(bridge).all():
                    raise ValueError('Invalid generated bridge')
                start=len(full)
                full=np.concatenate((full,bridge,placed),axis=0)
                segments.extend([dict(kind='bridge',source=order[(next_index-1)%len(order)],target=target,start=start,end=start+bridge_frames),
                                 dict(kind='primitive',action=target,start=start+bridge_frames,end=len(full))])
                edges.append(dict(start=start,end=start+bridge_frames,placement=placement))
                print(f'{group}: bridge {next_index}, target={target}, duration={(len(full)-1)/30:.2f}s',flush=True)
                next_index+=1
                if next_index>100:
                    raise RuntimeError('Unexpected number of segments')
            full,normalization=normalize_root(full)
            name='long_'+group
            feature_path=a.output_dir/'features'/group/(name+'.npz')
            feature_path.parent.mkdir(parents=True)
            np.savez_compressed(feature_path,features=full,fps=np.float32(30))
            native_path=a.output_dir/'soma77'/group/(name+'.npz')
            features_to_kimodo_npz(full,native_path,num_joints=30,fps=30,kimodo_root=a.kimodo_root,export_num_joints=77)
            record=dict(group=group,name=name,frames=len(full),fps=30,duration_s=(len(full)-1)/30,
                        order=[s['action'] for s in segments if s['kind']=='primitive'],segments=segments,edges=edges,
                        feature=str(feature_path.resolve()),native=str(native_path.resolve()),normalization=normalization,
                        sources={key:{k:v for k,v in source.items() if k!='features'} for key,source in clips.items()})
            manifest['clips'].append(record)
            (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({r['group']:r['duration_s'] for r in manifest['clips']}),flush=True)


if __name__=='__main__':
    main()
