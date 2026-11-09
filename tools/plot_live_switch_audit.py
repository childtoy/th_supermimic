#!/usr/bin/env python3
"""Plot paired live-switch controls and independent WM forecasts."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    d=json.loads((a.directory/'result.json').read_text())
    rows=d['rows'];phases=d['phase_seconds']
    pairs=sorted({(r['source'],r['target']) for r in rows})
    short={'boxing':'Box','traffic_control1':'T1','traffic_control2':'T2'}
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,mode in zip(axes[0],['policy','direct_cut']):
        grid=np.array([[next(r['transition_score'] for r in rows if r['mode']==mode and
            (r['source'],r['target'])==pair and r['phase_s']==phase) for phase in phases] for pair in pairs])
        im=ax.imshow(grid,vmin=.4,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(len(phases)),[f'{x:g}' for x in phases]);ax.set_xlabel('Source command time (s)')
        ax.set_yticks(range(len(pairs)),[short[s]+' -> '+short[t] for s,t in pairs])
        ax.set_title(mode+' | transition score')
        for i,j in np.ndindex(grid.shape):
            ax.text(j,i,f'{grid[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if grid[i,j]<.75 else 'black')
        fig.colorbar(im,ax=ax,shrink=.8)
    ax=axes[1,0];modes=list(d['summary']);x=np.arange(len(modes));n=len(rows)//len(modes)
    ax.bar(x-.18,[d['summary'][m]['completed'] for m in modes],.36,label='Hard completion')
    ax.bar(x+.18,[d['summary'][m]['quality_pass'] for m in modes],.36,label='Quality screen >= 0.8')
    ax.set_xticks(x,['RMAI bridge','Zero action','Direct cut']);ax.set_ylim(0,n+3);ax.set_ylabel(f'Count / {n}');ax.legend()
    ax=axes[1,1]
    for mode,color in [('policy','tab:blue'),('zero_after_command','tab:red')]:
        rr=[r for r in d['forecasts'] if r['kind']=='bridge' and r['mode']==mode]
        ax.scatter([r['sim_return'] for r in rr],[r['wm_return'] for r in rr],label=mode,s=22,color=color,alpha=.7)
    ax.plot([0,1],[0,1],'k--',linewidth=1);ax.set(xlabel='Actual bridge score',ylabel='Free WM predicted score',xlim=(0,1),ylim=(0,1));ax.legend()
    fig.suptitle('Frozen long-motion policy: uninterrupted A -> command -> full B\nB entry fixed at 0; one command per trial; deterministic simulation')
    for ext in ['png','pdf']:
        fig.savefig(a.directory/('audit.'+ext),dpi=180)
    print(a.directory/'audit.png')

if __name__=='__main__':
    main()
