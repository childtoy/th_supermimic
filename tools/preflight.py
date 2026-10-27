#!/usr/bin/env python3
"""Inspect versions, GPU runtime and prepared model identity before server training."""
import argparse
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
p = argparse.ArgumentParser()
p.add_argument('--robot', choices=['k1','g1'], required=True)
p.add_argument('--require-cuda', action='store_true')
a = p.parse_args()
from th_supermimic.robots import get_robot
spec = get_robot(a.robot)
for package in ['torch','mujoco','mujoco-warp','warp-lang']:
    try: print(f'{package}: {version(package)}')
    except PackageNotFoundError: print(f'{package}: missing')
import torch
import mujoco
print(f'CUDA available: {torch.cuda.is_available()}, torch CUDA runtime: {torch.version.cuda}')
if a.require_cuda and not torch.cuda.is_available():
    raise SystemExit('CUDA is required for the selected server training')
if torch.cuda.is_available():
    # Force runtime initialization rather than reporting only torch build metadata.
    x = torch.ones(8, device='cuda'); print(f'GPU: {torch.cuda.get_device_name()}, tensor sum: {x.sum().item()}')
model = mujoco.MjModel.from_xml_path(str(ROOT/'assets'/a.robot/'robot.xml'))
assert model.nq==7+spec.num_joints and model.nu==spec.num_joints
for i,n in enumerate(spec.joint_names):
    assert model.actuator(i).name==n
print(f'{a.robot}: {spec.num_joints} joints; nq={model.nq}, nv={model.nv}; asset checks passed')
