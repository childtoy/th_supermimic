#!/usr/bin/env python3
"""Render separated reference/actual views for qualitative flow review."""
import os,json,sys
from pathlib import Path
os.environ['MUJOCO_GL']='egl';os.environ['TH_SUPERMIMIC_ROBOT']='k1'
import numpy as np
import mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/long_switch_live_audit_54_20261104'
out=ROOT/'reports/long_switch_20261104/flow_review';out.mkdir(exist_ok=True)
d=json.loads((run/'result.json').read_text());v=np.load(run/'videos.npz')
model=mujoco.MjModel.from_xml_path(str(ROOT/'assets/k1/scene.xml'))
model.vis.global_.offwidth=480;model.vis.global_.offheight=360
model.vis.headlight.ambient[:]=.45;model.vis.headlight.diffuse[:]=.55
renderer=mujoco.Renderer(model,height=320,width=320);data=mujoco.MjData(model)
camera=mujoco.MjvCamera();camera.distance=3.2;camera.azimuth=135;camera.elevation=-12
try:
 for case in ['boxing__to__traffic_control1__t01500','traffic_control1__to__traffic_control2__t00500']:
    cols={d['rows'][int(i)]['mode']:col for col,i in enumerate(v['trial_indices']) if d['rows'][int(i)]['case']==case}
    row=next(r for r in d['rows'] if r['case']==case and r['mode']=='policy');cmd=round(row['command_time_s']*50)
    panels=[('Bridge reference','reference',cols['policy']),('Bridge actual','policy',cols['policy']),('Direct-cut actual','policy',cols['direct_cut'])]
    snapshots=[];times=[-.8,-.2,0,.2,.6,1.,1.4,1.8,2.4,2.6,3.,3.4]
    with imageio.get_writer(out/(case+'.mp4'),fps=25,codec='libx264',macro_block_size=None,quality=8) as writer:
     for idx in range(cmd-50,cmd+176,2):
      imgs=[]
      for label,key,col in panels:
       data.qpos[:]=v[key][idx,col];mujoco.mj_forward(model,data)
       camera.lookat[:]=[data.qpos[0],data.qpos[1],.75]
       renderer.update_scene(data,camera=camera)
       im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
       draw.rectangle((0,0,320,38),fill='white');draw.text((5,3),label,fill='black');draw.text((5,19),f'command offset {(idx-cmd)/50:+.2f}s',fill='black')
       imgs.append(np.array(im))
      frame=np.concatenate(imgs,axis=1);writer.append_data(frame)
      if any(abs((idx-cmd)/50-t)<.011 for t in times):snapshots.append(frame)
    # One triptych per row: reference / bridge actual / direct actual.
    Image.fromarray(np.concatenate(snapshots,axis=0)).save(out/(case+'__storyboard.png'))
    print(case,flush=True)
finally:renderer.close()
