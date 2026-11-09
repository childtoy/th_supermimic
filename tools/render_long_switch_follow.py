#!/usr/bin/env python3
"""Render recorded bridge + full canonical B continuation without resimulation."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--cases',nargs='+',required=True)
    a=p.parse_args()
    os.environ['TH_SUPERMIMIC_ROBOT']='k1'
    os.environ.setdefault('MUJOCO_GL','egl')
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import render_comparison
    result=json.loads((a.directory/'result_full_follow.json').read_text())
    first=np.load(a.directory/'result_full_follow_traces.npz')
    follow=np.load(a.directory/'full_follow_traces.npz')
    lookup={r['case']:r for r in result['full_target_follow']['rows']}
    for name in a.cases:
        index=next(i for i,c in enumerate(result['cases']) if c['name']==name)
        case=result['cases'][index];row=lookup[name]
        if row['failed_before_handoff']:
            raise ValueError('No successful handoff for selected case')
        steps=row['steps']
        assert int(follow['alive'][:steps,index].sum())==steps
        reference=np.concatenate((first['ref'][:,index],follow['reference'][:steps,index]))
        policy=np.concatenate((first['sim'][:,index],follow['policy'][:steps,index]))
        first_length=len(first['ref'])
        bridge_end=round((case['context_frames']+case['bridge_frames'])*50/30)-round((case['context_frames']-1)*50/30)
        title=f'{case["source"]} -> {case["target"]} @ {case["phase_s"]:g}s'
        labels=[title+' | bridge' if t<bridge_end else (
            'B entry (local reference)' if t<first_length else 'Original trained full B | no state/history reset') for t in range(len(policy))]
        output=a.directory/'videos_full'/(name+'.mp4')
        if output.exists():
            raise FileExistsError(output)
        render_comparison(reference,policy,output,root/'assets/k1/scene.xml',clip_name=title,
                          width=480,height=288,fps=5,global_ghost=True,frame_labels=labels)
        print(output,flush=True)


if __name__=='__main__':
    main()
