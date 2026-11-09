#!/usr/bin/env python3
"""Import the three named K1 mimic CSVs with explicit source convention checks."""
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,default=Path('../ai_sapiens'));p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 import numpy as np
 import yaml
 cfgpath=a.source_root/'ai_sapiens_sim2real/config/k1_config.yaml'
 cfg=yaml.safe_load(cfgpath.read_text());spec=json.loads((Path(__file__).resolve().parents[1]/'src/th_supermimic/robots/k1.json').read_text())
 assert cfg['robot_joint_order']==spec['joint_names'],'CSV fallback joint order differs'
 assert cfg['mimic_defaults']['fps']==50
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 records=[];staged=[]
 for name in ['dance1','dance2','squat']:
  path=a.source_root/f'ai_sapiens_sim2real/assets/k1/mimic/{name}/params/{name}.csv'
  q=np.loadtxt(path,delimiter=',');assert q.ndim==2 and q.shape[1]==30 and np.isfinite(q).all()
  err=float(np.max(np.abs(np.linalg.norm(q[:,3:7],axis=1)-1)))
  assert err<1e-3,('Invalid xyzw quaternion norm',err)
  records.append(dict(name=name,source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
   source_frames=len(q),fps=50,duration_s=(len(q)-1)/50,quaternion_norm_max_error=err,root_height_range_m=[float(q[:,2].min()),float(q[:,2].max())]))
  staged.append((path,name))
 a.output_dir.mkdir(parents=True)
 for path,name in staged:shutil.copyfile(path,a.output_dir/(name+'.csv'))
 manifest=dict(schema_version=1,robot='k1',fps=50,groups=['ai_sapiens'],clips=records,
  convention='Unmodified CSV: xyz + xyzw + robot_joint_order; verified against ai_sapiens motion_reference.cpp and k1_config.yaml. No retarget, resampling or transition generation.',
  source_config_sha256=hashlib.sha256(cfgpath.read_bytes()).hexdigest(),joint_names=spec['joint_names'])
 (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
