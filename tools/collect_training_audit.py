#!/usr/bin/env python3
"""Archive exact checkpoints, references, models and logs for offline investigation.

Read-only with respect to training artifacts. Does not load or modify checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
from pathlib import Path
import platform
import subprocess
import tarfile
import time

DEFAULT_GROUPS=('all','boxing','traffic_control1','traffic_control2')


def collect_files(root,groups,steps):
    selected=set()
    optional_missing=[]
    def add_tree(path):
        if not path.is_dir():
            raise FileNotFoundError(f'Missing required directory: {path}')
        for p in path.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
                selected.add(p)
    for group in groups:
        run=root/'runs'/f'k1_{group}'
        for name in ['last.pt','run_config.json']:
            p=run/name
            if not p.is_file():
                raise FileNotFoundError(f'Missing required artifact: {p}')
            selected.add(p)
        for step in steps:
            p=run/f'ckpt_{step:06d}.pt'
            if p.is_file():selected.add(p)
            else:optional_missing.append(str(p.relative_to(root)))
        for p in run.iterdir():
            if p.is_file() and (p.suffix in ('.json','.csv') or p.name.startswith('events.out.tfevents')):
                selected.add(p)
        for p in run.rglob('events.out.tfevents*'):
            if p.is_file():selected.add(p)
        log=root/'logs'/f'k1_{group}.log'
        if log.is_file():selected.add(log)
        motion=root/'data/k1'/group
        manifest=json.loads((motion/'manifest.json').read_text())
        if manifest['robot']!='k1' or manifest['fps']!=50:
            raise ValueError(f'Unexpected reference manifest: {motion}')
        selected.add(motion/'manifest.json')
        for clip in manifest['clips']:
            name=clip['name']
            if Path(name).name!=name:
                raise ValueError(f'Invalid clip name: {name}')
            csv=motion/(name+'.csv')
            if not csv.is_file():raise FileNotFoundError(csv)
            selected.add(csv)
    add_tree(root/'assets/k1')
    add_tree(root/'src/th_supermimic')
    add_tree(root/'tools')
    add_tree(root/'scripts')
    for name in ['pyproject.toml','README.md','UPSTREAM.json','NOTICE','LICENSE']:
        p=root/name
        if p.is_file():selected.add(p)
    for p in selected:
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError(f'Expected a regular repository-local artifact, got {p}')
    return sorted(selected),optional_missing


def build_archive(root,output,groups=DEFAULT_GROUPS,steps=(2000,4000)):
    root=root.resolve()
    files,missing=collect_files(root,groups,steps)
    if output.resolve() in {p.resolve() for p in files}:
        raise ValueError('Archive output must not overwrite an input artifact')
    output.parent.mkdir(parents=True,exist_ok=True)
    total=sum(p.stat().st_size for p in files)
    print(f'Packing {len(files)} files, {total/1024**3:.2f} GiB before compression',flush=True)
    if missing:print('Optional checkpoints missing: '+', '.join(missing),flush=True)
    versions={}
    for package in ['torch','numpy','mujoco','mujoco-warp','warp-lang']:
        try:versions[package]=version(package)
        except PackageNotFoundError:versions[package]=None
    def git(*args):
        try:
            r=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,timeout=10)
            return r.stdout if r.returncode==0 else None
        except (OSError,subprocess.TimeoutExpired):return None
    report=dict(schema=1,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                source_root=str(root),groups=list(groups),steps=list(steps),missing_optional=missing,
                versions=versions,python=platform.python_version(),platform=platform.platform(),
                collection_git_head=git('rev-parse','HEAD'),collection_git_diff=git('diff','HEAD'),
                note='Collection revision may differ from training revision. Checkpoints are copied unchanged.',files=[])
    tmp=output.with_name(output.name+'.tmp')
    try:
        with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
            for i,p in enumerate(files,1):
                before=p.stat();digest=hashlib.sha256()
                # Hash while tarfile streams the file; avoid allocating full checkpoints in RAM.
                class HashReader:
                    def __init__(self,stream):self.stream=stream
                    def read(self,size=-1):
                        data=self.stream.read(size);digest.update(data);return data
                info=tar.gettarinfo(str(p),arcname='training_audit/'+p.relative_to(root).as_posix())
                info.uid=info.gid=0;info.uname=info.gname=''
                with p.open('rb') as stream:tar.addfile(info,HashReader(stream))
                after=p.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise RuntimeError(f'Input changed during collection; stop training before retrying: {p}')
                report['files'].append(dict(path=p.relative_to(root).as_posix(),bytes=before.st_size,
                                           sha256=digest.hexdigest()))
                if p.suffix=='.pt' or i==len(files):
                    print(f'[{i}/{len(files)}] {p.relative_to(root)}',flush=True)
            payload=(json.dumps(report,indent=2)+'\n').encode()
            info=tarfile.TarInfo('training_audit/audit_manifest.json');info.size=len(payload)
            tar.addfile(info,io.BytesIO(payload))
        tmp.replace(output)
    finally:
        if tmp.exists():tmp.unlink()
    print(f'Bundle ready: {output} ({output.stat().st_size/1024**3:.2f} GiB)',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path)
    p.add_argument('--groups',nargs='+',choices=DEFAULT_GROUPS,default=list(DEFAULT_GROUPS))
    p.add_argument('--steps',nargs='*',type=int,default=[2000,4000])
    a=p.parse_args()
    build_archive(a.repo_root,a.output or a.repo_root/'exports/k1_training_audit.tar.gz',a.groups,a.steps)
