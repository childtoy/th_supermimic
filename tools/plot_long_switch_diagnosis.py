#!/usr/bin/env python3
"""Plot frozen-policy switch outcomes and WM forecast error from saved results."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((a.directory/'result.json').read_text())
    rows=[r for r in result['rows'] if r['kind']=='bridge' and not r.get('control')]
    cases=result['cases'];groups=['boxing','traffic_control1','traffic_control2']
    short=dict(boxing='Box',traffic_control1='T1',traffic_control2='T2')
    pairs=[(s,t) for s in groups for t in groups if s!=t]
    phases=sorted({r['phase_s'] for r in rows})
    image=np.full((len(pairs),len(phases)),np.nan)
    for r in rows:
        image[pairs.index((r['source'],r['target'])),phases.index(r['phase_s'])]=r['sim_return']
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    ax=axes[0,0]
    art=ax.imshow(image,aspect='auto',vmin=min(0,float(np.nanmin(image))),vmax=1,cmap='viridis')
    ax.set_xticks(range(len(phases)),[f'{v:g}s' for v in phases]);ax.set_yticks(range(len(pairs)),[f'{short[s]} → {short[t]}' for s,t in pairs])
    for r in rows:
        y=pairs.index((r['source'],r['target']));x=phases.index(r['phase_s'])
        flags=(' R' if r['sim_failed'] else '')+(' W' if r['wm_failed'] else '')
        ax.text(x,y,f'{r["sim_return"]:.2f}{flags}',ha='center',va='center',color='white',fontsize=9)
    ax.set_xlabel('Source phase');ax.set_title('Actual bridge score (R: actual failure, W: predicted failure)')
    fig.colorbar(art,ax=ax,label='Normalized mean bridge reward')
    ax=axes[0,1]
    colors=dict(boxing='tab:orange',traffic_control1='tab:blue',traffic_control2='tab:green')
    for group in groups:
        rs=[r for r in rows if r['source']==group]
        ax.scatter([r['wm_return'] for r in rs],[r['sim_return'] for r in rs],c=colors[group],label=short[group])
    low=min(0,min(min(r['wm_return'],r['sim_return']) for r in rows));high=max(1,max(max(r['wm_return'],r['sim_return']) for r in rows))
    ax.plot([low,high],[low,high],'k--',lw=1);ax.set_xlim(low-.03,high+.03);ax.set_ylim(low-.03,high+.03)
    ax.set_xlabel('WM-predicted bridge reward');ax.set_ylabel('Actual bridge reward');ax.legend(title='Source');ax.set_title('Closed-loop WM calibration')
    worst=max(range(len(rows)),key=lambda i:abs(rows[i]['wm_return']-rows[i]['sim_return']))
    name=rows[worst]['case'];index=next(i for i,c in enumerate(cases) if c['name']==name)
    trace=np.load(a.directory/'traces.npz');time=(np.arange(len(trace['sim_reward']))+1)/50
    ax=axes[1,0]
    ax.plot(time,trace['sim_reward'][:,index],label='Actual');ax.plot(time,trace['wm_reward'][:,index],label='WM',alpha=.8)
    case=cases[index];anchor=round((case['context_frames']-1)*50/30)
    start=(round(case['context_frames']*50/30)-anchor)/50
    end=(round((case['context_frames']+case['bridge_frames'])*50/30)-anchor)/50
    ax.axvspan(start,end,color='gray',alpha=.15,label='Bridge')
    ax.set_xlabel('Time after reference switch (s)');ax.set_ylabel('Normalized step reward');ax.legend()
    ax.set_title(f'Largest score mismatch: {short[case["source"]]} → {short[case["target"]]} @ {case["phase_s"]:g}s')
    ax=axes[1,1]
    for group in groups:
        rs=[r for r in rows if r['source']==group]
        ax.scatter([r['source_reference_joint_jump_rad'] for r in rs],[r['sim_return'] for r in rs],c=colors[group],label=short[group])
    ax.set_xlabel('Source reference joint discontinuity (RMS rad)');ax.set_ylabel('Actual bridge reward')
    ax.set_title('Boundary discontinuity vs. execution quality');ax.legend(title='Source')
    fig.suptitle('Long-motion switching: fixed base policy, 18 untrained RMAI bridges',fontsize=14)
    fig.savefig(a.directory/'diagnosis.png',dpi=180)
    fig.savefig(a.directory/'diagnosis.pdf')
    print(a.directory/'diagnosis.png')


if __name__=='__main__':
    main()
