#!/usr/bin/env python3
"""Render an uninterrupted recorded audit trial, including the source prefix."""
import argparse
import json
import os
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--case',required=True)
    p.add_argument('--modes',nargs='+',default=['policy','direct_cut'])
    p.add_argument('--window',nargs=2,type=float,help='Seconds before/after command')
    a=p.parse_args()
    os.environ['TH_SUPERMIMIC_ROBOT']='k1'
    os.environ.setdefault('MUJOCO_GL','egl')
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import render_comparison
    result=json.loads((a.directory/'result.json').read_text())
    data=np.load(a.directory/'videos.npz')
    for mode in a.modes:
        trial=next(i for i,r in enumerate(result['rows']) if r['case']==a.case and r['mode']==mode)
        col=list(data['trial_indices']).index(trial)
        row=result['rows'][trial]
        n=int(data['alive'][:,col].sum())
        assert data['alive'][:n,col].all()
        title=f'{a.case} | {mode}'
        lo=0 if a.window is None else max(0,round((row['command_time_s']-a.window[0])*50))
        hi=n if a.window is None else min(n,round((row['command_time_s']+a.window[1])*50))
        labels=[title+f' | command offset {t/50-row["command_time_s"]:+.2f}s' for t in range(lo,hi)]
        suffix='' if a.window is None else '__window'
        output=a.directory/'videos'/(a.case+'__'+mode+suffix+'.mp4')
        if output.exists():
            raise FileExistsError(output)
        render_comparison(data['reference'][lo:hi,col],data['policy'][lo:hi,col],output,
            root/'assets/k1/scene.xml',clip_name=title,width=480,height=288,fps=5,
            global_ghost=True,frame_labels=labels)
        print(output,flush=True)

if __name__=='__main__':
    main()
