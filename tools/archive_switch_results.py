#!/usr/bin/env python3
"""Archive recorded motion smoothness with explicit derivative conventions."""
import json,csv,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/long_switch_live_audit_54_20261104'
out=ROOT/'reports/long_switch_20261104'
out.mkdir(parents=True,exist_ok=True)
d=json.loads((run/'result.json').read_text());v=np.load(run/'videos.npz')
rows=[]
fig,axes=plt.subplots(4,2,figsize=(13,12),layout='constrained')
for col,trial in enumerate(v['trial_indices']):
    r=d['rows'][int(trial)];n=int(v['alive'][:,col].sum());t=np.arange(n)/50-r['command_time_s']
    for kind in ['policy','reference']:
        q=v[kind][:n,col,7:]
        # Derive position-based jerk identically for reference and actual.
        vel=savgol_filter(q,11,5,deriv=1,delta=.02,axis=0)
        acc=savgol_filter(q,11,5,deriv=2,delta=.02,axis=0)
        jerk=savgol_filter(q,11,5,deriv=3,delta=.02,axis=0)
        raw_jerk=np.diff(q,n=3,axis=0)/.02**3
        jr=np.sqrt(np.mean(jerk**2,axis=1));ar=np.sqrt(np.mean(acc**2,axis=1))
        for name,left,right in [('source',-2,-.25),('command',-.2,.5),('transition',0,1.54),('handoff',2.34,2.74)]:
            if name=='handoff' and r['mode']=='direct_cut':continue
            mask=(t>=left)&(t<right)
            rm=(t[3:]>=left)&(t[3:]<right)
            rows.append(dict(case=r['case'],mode=r['mode'],signal=kind,window=name,
                jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk[mask]**2))),
                jerk_peak_joint_rad_s3=float(np.max(np.abs(jerk[mask]))),
                acceleration_rms_rad_s2=float(np.sqrt(np.mean(acc[mask]**2))),
                raw_jerk_rms_rad_s3=float(np.sqrt(np.mean(raw_jerk[rm]**2)))))
        ax=axes.flat[col];mask=(t>=-2)&(t<=4)
        ax.plot(t[mask],jr[mask],label=kind)
        ax.axvline(0,color='k',ls='--',lw=.8)
        if r['mode']=='policy':ax.axvline(2.54,color='gray',ls=':',lw=.8)
        ax.set_title(r['case']+' | '+r['mode'],fontsize=8)
        ax.set(xlabel='Time from command (s)',ylabel='Joint RMS jerk (rad/s³)');ax.legend(fontsize=7)
fig.suptitle('Recorded 50 Hz joint positions | Savitzky-Golay 11 samples / degree 5 / third derivative\n8 recorded trials only; command at 0, canonical handoff at +2.54 s for bridge')
for ext in ['png','pdf']:fig.savefig(out/('jerk_windows.'+ext),dpi=160)
fig2,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
cases=list(dict.fromkeys(d['rows'][int(i)]['case'] for i in v['trial_indices']))
for ax,case in zip(axs.flat,cases):
    for col,trial in enumerate(v['trial_indices']):
        r=d['rows'][int(trial)]
        if r['case']!=case:continue
        n=int(v['alive'][:,col].sum());t=np.arange(n)/50-r['command_time_s']
        j=savgol_filter(v['policy'][:n,col,7:],11,5,deriv=3,delta=.02,axis=0)
        mask=(t>=-2)&(t<=4)
        ax.plot(t[mask],np.sqrt(np.mean(j[mask]**2,axis=1)),label=r['mode'])
    ax.axvline(0,color='k',ls='--',lw=.8);ax.axvline(2.54,color='gray',ls=':',lw=.8)
    ax.set(title=case,xlabel='Time from command (s)',ylabel='Actual joint RMS jerk (rad/s³)');ax.legend()
fig2.suptitle('Actual robot smoothness: bridge vs direct cut | identical 50 Hz derivative settings')
for ext in ['png','pdf']:fig2.savefig(out/('actual_jerk_comparison.'+ext),dpi=160)
with (out/'smoothness.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name in ['result.json','trials.csv','audit.png','audit.pdf']:
    shutil.copy2(run/name,out/name)
manifest=dict(checkpoint_sha256=d['checkpoint_sha256'],iteration=d['iteration'],
    source_run=str(run.relative_to(ROOT)),recorded_trials=[int(i) for i in v['trial_indices']],
    sample_hz=50,derivative='Position-based SG window 11 samples (0.22s), polynomial degree 5; derivatives over full continuous trace before slicing; also raw third differences',
    limitations=['Smoothness covers eight recorded policy/direct-cut trials, not all 162.',
    'No measured torque traces are available; action differences are not actual torque.',
    'Source window is continuation before command, not a matched counterfactual run.',
    'Reference switch discontinuities affect numerical jerk; filtering spreads peaks over nearby samples.',
    'No physical safety threshold or deployment validation is implied.'],
    files={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ['result.json','trials.csv','videos.npz','wm_audit_inputs.pt']})
(out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out)
