"""MuJoCo qpos/qvel collectors; joint columns are mapped by name."""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
import torch
import mujoco

SUPPORTED_BACKENDS=("mujoco","mjwarp")


class CPUBackend:
    def __init__(self, model_path, spec, num_envs, device='cpu'):
        if torch.device(device).type != 'cpu':
            raise ValueError('CPUBackend uses CPU tensors')
        self.spec, self.num_envs, self.device = spec, num_envs, torch.device('cpu')
        self.models = [mujoco.MjModel.from_xml_path(str(model_path)) for _ in range(num_envs)]
        self.model = self.models[0]
        self.data = [mujoco.MjData(m) for m in self.models]
        self.qadr = [self.model.jnt_qposadr[self.model.joint(n).id] for n in spec.joint_names]
        self.vadr = [self.model.jnt_dofadr[self.model.joint(n).id] for n in spec.joint_names]
        self.ctrladr = [self.model.actuator(n).id for n in spec.joint_names]
        self.com0 = self.model.body_ipos[self.model.body(spec.context_body).id].copy()
        self.substeps = round(.02 / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, .02):
            raise ValueError('Model timestep must divide 0.02 seconds')

    def randomize(self, ids, friction, com):
        for row, env in enumerate(ids.tolist()):
            m = self.models[env]
            m.geom_friction[:,0] = float(friction[row])
            m.body_ipos[m.body(self.spec.context_body).id] = self.com0 + com[row].cpu().numpy()
            mujoco.mj_setConst(m, self.data[env])

    def reset(self, ids, qpos, qvel):
        for i, env in enumerate(ids.tolist()):
            d = self.data[env]
            mujoco.mj_resetData(self.models[env], d)
            d.qpos[:] = qpos[i].cpu().numpy()
            d.qvel[:] = qvel[i].cpu().numpy()
            d.ctrl[self.ctrladr] = qpos[i, self.qadr].cpu().numpy()
            mujoco.mj_forward(self.models[env], d)

    def step(self, target):
        for i,d in enumerate(self.data):
            d.ctrl[self.ctrladr] = target[i].cpu().numpy()
            mujoco.mj_step(self.models[i],d,nstep=self.substeps)

    def read(self):
        return (torch.from_numpy(np.stack([d.qpos.copy() for d in self.data])).float(),
                torch.from_numpy(np.stack([d.qvel.copy() for d in self.data])).float())

    def close(self):
        pass


class WarpBackend:
    """Own Warp state buffers and capture only the physics substeps."""
    def __init__(self, model_path, spec, num_envs, device='cuda:0'):
        import warp as wp
        import mujoco_warp as mw
        if torch.device(device).type != 'cuda':
            raise ValueError('MuJoCo Warp training requires a CUDA device')
        self.wp, self.mw, self.spec = wp, mw, spec
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.qadr = [self.model.jnt_qposadr[self.model.joint(n).id] for n in spec.joint_names]
        self.vadr = [self.model.jnt_dofadr[self.model.joint(n).id] for n in spec.joint_names]
        self.ctrladr = [self.model.actuator(n).id for n in spec.joint_names]
        self.substeps = round(.02/self.model.opt.timestep)
        if not np.isclose(self.substeps*self.model.opt.timestep,.02):
            raise ValueError('Model timestep must divide 0.02 seconds')
        wp.init()
        self.torch_stream = torch.cuda.Stream(device=self.device)
        self.stream = wp.stream_from_torch(self.torch_stream)
        with self._scope():
            self.m = mw.put_model(self.model)
            if hasattr(self.m.opt,'warn_overflow'):
                self.m.opt.warn_overflow = 0
            self.d = mw.make_data(self.model,nworld=num_envs,nconmax=128,njmax=512)
            self.qpos = wp.to_torch(self.d.qpos)
            self.qvel = wp.to_torch(self.d.qvel)
            self.ctrl = wp.to_torch(self.d.ctrl)
            self.friction = torch.tensor(self.model.geom_friction,device=device,dtype=torch.float32).repeat(num_envs,1,1)
            self.ipos = torch.tensor(self.model.body_ipos,device=device,dtype=torch.float32).repeat(num_envs,1,1)
            self.ipos0 = self.ipos.clone()
            self.m.geom_friction = wp.from_torch(self.friction,dtype=wp.vec3)
            self.m.body_ipos = wp.from_torch(self.ipos,dtype=wp.vec3)
            self.com_id = self.model.body(spec.context_body).id
            mw.set_const(self.m,self.d)
            mw.forward(self.m,self.d)
            # Compile before capture. Restore a valid reset before policy stepping.
            for _ in range(self.substeps):
                mw.step(self.m,self.d)
            with wp.ScopedCapture() as capture:
                for _ in range(self.substeps):
                    mw.step(self.m,self.d)
            self.graph = capture.graph

    @contextmanager
    def _scope(self):
        current=torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(current)
        try:
            with torch.cuda.stream(self.torch_stream), self.wp.ScopedDevice(str(self.device)), self.wp.ScopedStream(self.stream):
                yield
        finally:
            current.wait_stream(self.torch_stream)

    def randomize(self, ids, friction, com):
        with self._scope():
            self.friction[ids,:,0] = friction[:,None]
            self.ipos[ids,self.com_id] = self.ipos0[ids,self.com_id] + com
            self.mw.set_const(self.m,self.d)

    def reset(self, ids, qpos, qvel):
        with self._scope():
            self.qpos[ids] = qpos
            self.qvel[ids] = qvel
            self.ctrl[ids] = qpos[:,self.qadr]
            self.wp.to_torch(self.d.qacc_warmstart)[ids] = 0
            if hasattr(self.d,'time'):
                self.wp.to_torch(self.d.time)[ids] = 0
            self.mw.forward(self.m,self.d)

    def step(self,target):
        with self._scope():
            self.ctrl[:,self.ctrladr] = target
            self.wp.capture_launch(self.graph)

    def read(self):
        return self.qpos.clone(),self.qvel.clone()

    def close(self):
        self.wp.synchronize_device(str(self.device))
