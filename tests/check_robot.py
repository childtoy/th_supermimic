"""Executed in a fresh process per robot to exercise per-run selection."""
from pathlib import Path
import sys
import numpy as np
import mujoco
import torch
from th_supermimic.robots import ROBOT
from th_supermimic.tasks.motion_tracking.humanoid.supermimic.kinematics import HumanoidKinematics
from th_supermimic.tasks.motion_tracking.humanoid.supermimic.motion import MotionLib
from th_supermimic.tasks.motion_tracking.humanoid.supermimic.mdp import MDP
from th_supermimic.rl.algos.supermimic.geometry import quat_to_mat
from th_supermimic.rl.algos.supermimic.networks import WorldModel

root=Path(__file__).resolve().parents[1]
path=root/'assets'/ROBOT.name/'robot.xml'
model=mujoco.MjModel.from_xml_path(str(path)); data=mujoco.MjData(model)
kin=HumanoidKinematics(str(path),'cpu')
n=ROBOT.num_joints
rng=np.random.default_rng(42)
for _ in range(4):
    q=np.asarray(ROBOT.default_joint_pos)+rng.normal(0,.05,n)
    qd=rng.normal(0,.1,n)
    data.qpos[:3]=[.2,-.3,ROBOT.root_height]
    data.qpos[3:7]=[.98,.05,.1,.12];data.qpos[3:7]/=np.linalg.norm(data.qpos[3:7])
    data.qvel[:6]=rng.normal(0,.1,6)
    for i,name in enumerate(ROBOT.joint_names):
        j=model.joint(name).id
        data.qpos[model.jnt_qposadr[j]]=q[i];data.qvel[model.jnt_dofadr[j]]=qd[i]
    mujoco.mj_forward(model,data)
    R=quat_to_mat(torch.tensor(data.qpos[3:7],dtype=torch.float32))[None]
    # MuJoCo freejoint angular velocity is body-local; FK uses world angular velocity.
    state=(torch.tensor(data.qpos[:3],dtype=torch.float32)[None],R,
           torch.tensor(data.qvel[:3],dtype=torch.float32)[None],
           (R@torch.tensor(data.qvel[3:6],dtype=torch.float32)[None,:,None]).squeeze(-1),
           torch.tensor(q,dtype=torch.float32)[None],torch.tensor(qd,dtype=torch.float32)[None])
    p,r,v,w=kin.forward(*state)
    for i,name in enumerate(ROBOT.tracked_bodies):
        b=model.body(name).id
        np.testing.assert_allclose(p[0,i].numpy(),data.xpos[b],atol=2e-6)
        np.testing.assert_allclose(r[0,i].numpy(),data.xmat[b].reshape(3,3),atol=2e-6)
        vel=np.empty(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,b,vel,0)
        # mj_objectVelocity linear part is at inertial COM; translate to body origin.
        origin_v=vel[3:]+np.cross(vel[:3], data.xpos[b]-data.xipos[b])
        np.testing.assert_allclose(v[0,i].numpy(),origin_v,atol=3e-6)
        np.testing.assert_allclose(w[0,i].numpy(),vel[:3],atol=3e-6)
# Differentiable motion/MDP and world-model gradients for both morphologies.
import tempfile
with tempfile.TemporaryDirectory() as temp:
    csv=Path(temp)/'clip.csv'
    row=np.r_[data.qpos[:3],data.qpos[[4,5,6,3]],q]
    np.savetxt(csv,np.tile(row,(80,1)),delimiter=',')
    lib=MotionLib([str(csv)],kin,'cpu',src_fps=50)
    mdp=MDP(kin,lib,torch.tensor(ROBOT.default_joint_pos))
    frames=torch.tensor([0]);context=torch.zeros(1,ROBOT.context_dim);context[:,0]=.75
    action=torch.zeros(1,n,requires_grad=True)
    assert mdp.obs(state,action,context,frames).shape==(1,3*n+12)
    assert mdp.goal(frames,10).shape==(1,(15+2*n)*10)
    wm=WorldModel(hidden=(32,32),num_joints=n,context_dim=ROBOT.context_dim)
    # Nonzero learned weights to verify action -> model -> analytic reward gradient.
    torch.nn.init.normal_(wm.net[-1].weight,std=.01)
    predicted=state
    for _ in range(4):
        predicted=wm.step(predicted,action,mdp.normalize_context(context))
    loss=-mdp.reward(predicted,frames,action,torch.zeros_like(action)).mean()
    loss.backward()
    assert action.grad is not None and torch.isfinite(action.grad).all() and action.grad.abs().sum()>0
print(f'{ROBOT.name}: FK positions/orientations/velocities and 4-step BPTT passed')
