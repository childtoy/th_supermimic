import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from th_supermimic.robots import ROBOT
from th_supermimic.sim.backends import CPUBackend,WarpBackend
from pathlib import Path
cpu=CPUBackend(ROOT/'assets'/ROBOT.name/'robot.xml',ROBOT,2)
gpu=WarpBackend(ROOT/'assets'/ROBOT.name/'robot.xml',ROBOT,2,'cuda:0')
q=torch.zeros(2,cpu.model.nq);v=torch.zeros(2,cpu.model.nv)
q[:,2]=ROBOT.root_height;q[:,3]=1;q[:,cpu.qadr]=torch.tensor(ROBOT.default_joint_pos)
for sim in [cpu,gpu]:
 ids=torch.arange(2,device=sim.device)
 sim.randomize(ids,torch.ones(2,device=sim.device),torch.zeros(2,3,device=sim.device))
 sim.reset(ids,q.to(sim.device),v.to(sim.device))
 sim.step(torch.tensor(ROBOT.default_joint_pos,device=sim.device).repeat(2,1))
a,av=cpu.read();b,bv=gpu.read()
print(ROBOT.name,'qpos max diff',(a-b.cpu()).abs().max().item(),'qvel max diff',(av-bv.cpu()).abs().max().item())
assert torch.allclose(a,b.cpu(),atol=1e-3,rtol=1e-3)
assert torch.allclose(av,bv.cpu(),atol=3e-2,rtol=3e-2)
for sim in [cpu,gpu]:sim.close()
