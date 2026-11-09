#!/usr/bin/env python3
"""Describe motion flow without assigning a naturalness score."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];run=root/'runs/long_switch_live_audit_54_20261104';out=root/'reports/long_switch_20261104/flow_review';out.mkdir(exist_ok=True)
d=json.loads((run/'result.json').read_text());v=np.load(run/'videos.npz');records=[]
fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for ax,case in zip(axs.flat,dict.fromkeys(d['rows'][int(i)]['case'] for i in v['trial_indices'])):
 for col,trial in enumerate(v['trial_indices']):
  r=d['rows'][int(trial)]
  if r['case']!=case:continue
  cmd=round(r['command_time_s']*50);n=int(v['alive'][:,col].sum());t=np.arange(1,n)/50-r['command_time_s']
  for key in ['reference','policy']:
   if key=='reference' and r['mode']=='direct_cut':continue
   q=v[key][:n,col,7:];speed=np.sqrt(np.mean((np.diff(q,axis=0)*50)**2,axis=1));mask=(t>=-1)&(t<=3.5)
   # Keep boundary discontinuities in data but exclude exact switch samples from flow plot.
   show=mask&(np.abs(t)>.04)&(np.abs(t-2.54)>.04)
   y=np.where(show,speed,np.nan);ax.plot(t[mask],y[mask],label=('bridge reference' if key=='reference' else r['mode']+' actual'))
   for lo,hi in [(-1,0),(0,.5),(.5,1),(1,1.5),(1.5,2),(2,2.5),(2.6,3.4)]:
    x=q[cmd+round(lo*50):cmd+round(hi*50)];s=np.sqrt(np.mean((np.diff(x,axis=0)*50)**2,axis=1))
    records.append(dict(case=case,mode=r['mode'],signal=key,start_s=lo,end_s=hi,mean_joint_rms_speed_rad_s=float(s.mean()),joint_pose_span_rms_rad=float(np.sqrt(np.mean(np.ptp(x,axis=0)**2)))))
 ax.axvline(0,color='k',ls='--');ax.axvline(1.54,color='gray',ls='--');ax.axvline(2.54,color='gray',ls=':')
 ax.set(title=case,xlabel='Time from command (s)',ylabel='Joint RMS speed (rad/s)');ax.legend(fontsize=7)
fig.suptitle('Motion flow descriptors, not naturalness scores | dashed 0=command, 1.54=B entry; dotted 2.54=handoff\nExact switch ±0.04 s omitted from plot to show motion within segments; raw traces retained')
fig.savefig(out/'flow_speed.png',dpi=160);fig.savefig(out/'flow_speed.pdf');(out/'flow_descriptors.json').write_text(json.dumps(records,indent=2)+'\n')
