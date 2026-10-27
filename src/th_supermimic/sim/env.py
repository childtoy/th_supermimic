"""Tracking lifecycle and pre-reset replay transitions owned by th_supermimic."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import torch
from th_supermimic.robots import ROBOT
from th_supermimic.rl.algos.supermimic.geometry import flatten_state,mat_to_quat,quat_to_mat,so3_exp
from th_supermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY,SUPERMIMIC_TRANSITION_KEY
from th_supermimic.tasks.motion_tracking.humanoid.supermimic.kinematics import HumanoidKinematics
from th_supermimic.tasks.motion_tracking.humanoid.supermimic.motion import MotionLib,BIN_FRAMES
from th_supermimic.tasks.motion_tracking.humanoid.supermimic.mdp import MDP
from .backends import CPUBackend,WarpBackend


@dataclass
class CollectorState:
    info: dict


class TrackingEnv:
    def __init__(self,assets_dir,motion_files,num_envs=4096,backend='mjwarp',device=None,
                 eval_mode=False,gamma=.99,max_episode_steps=500,seed=42,randomization=True):
        if backend not in ('mujoco','mjwarp'):
            raise ValueError('Choose mujoco or mjwarp')
        if num_envs < 1:
            raise ValueError('num_envs must be positive')
        self.spec,self.num_envs,self.backend_name=ROBOT,num_envs,backend
        self.device=torch.device(device or ('cuda:0' if backend=='mjwarp' else 'cpu'))
        self.eval_mode,self.gamma=eval_mode,gamma
        self.randomization=randomization and not eval_mode
        self.max_episode_steps=max_episode_steps
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        self.model_path=Path(assets_dir)/'robot.xml'
        self.kin=HumanoidKinematics(str(self.model_path),self.device)
        self.motion=MotionLib(motion_files,self.kin,self.device,src_fps=50)
        self.default_q=torch.tensor(ROBOT.default_joint_pos,device=self.device)
        self.supermimic_mdp=MDP(self.kin,self.motion,self.default_q)
        self.sim=(CPUBackend if backend=='mujoco' else WarpBackend)(self.model_path,ROBOT,num_envs,self.device)
        self.contract=dict(schema=2,robot=ROBOT.name,joint_names=list(ROBOT.joint_names),
            model_sha256=hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
            motions=[dict(name=Path(p).name,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in motion_files])
        self.alpha=torch.tensor([m['action_scale'] for m in ROBOT.motors],device=self.device)
        self.f=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.ep_step=torch.zeros_like(self.f)
        self.a_prev=torch.zeros(num_envs,ROBOT.num_joints,device=self.device)
        self.xi=torch.zeros(num_envs,ROBOT.context_dim,device=self.device)
        self.xi[:,0]=1. if eval_mode else .75
        # Torso CoM is a per-world startup context; joint offsets/friction resample per episode.
        if self.randomization:
            self.xi[:,1+ROBOT.num_joints:]=self.uniform(-.05,.05,(num_envs,3))
        self.start_bin=torch.zeros_like(self.f)
        self.bin_G=torch.zeros(self.motion.num_bins,device=self.device)
        self.bin_memory=torch.ones_like(self.bin_G)/self.motion.num_bins
        self.ep_ret=torch.zeros(num_envs,device=self.device)
        self.ep_disc=torch.zeros_like(self.ep_ret)
        self.state=None
        self.reset(torch.arange(num_envs,device=self.device))
        self._publish()

    def uniform(self,lo,hi,shape):
        return lo+(hi-lo)*torch.rand(shape,generator=self.rng,device=self.device)

    def sample_frames(self,count):
        soft=torch.softmax(-25*self.bin_G/(self.supermimic_mdp.max_step_reward/(1-self.gamma)),0)
        self.bin_memory.mul_(.999).add_(soft,alpha=.001)
        probs=.5/self.motion.num_bins+.4*soft+.1*self.bin_memory/self.bin_memory.sum()
        bins=torch.multinomial(probs,count,replacement=True,generator=self.rng)
        start=self.motion.bin_start[bins]
        end=self.motion.clip_end[self.motion.bin_clip[bins]]
        # A bin containing only the final frame has no outgoing transition.
        start=torch.minimum(start,end-1)
        upper=torch.minimum(start+BIN_FRAMES,end)
        frames=start+(self.uniform(0,1,(count,))*(upper-start)).long()
        return frames,bins

    def read(self):
        qpos,qvel=self.sim.read()
        rotation=quat_to_mat(qpos[:,3:7])
        return (qpos[:,:3],rotation,qvel[:,:3],
                (rotation@qvel[:,3:6,None]).squeeze(-1),
                qpos[:,self.sim.qadr],qvel[:,self.sim.vadr])

    def info(self):
        return dict(s=flatten_state(self.read()).detach().clone(),f=self.f.clone(),xi=self.xi.clone(),
                    a_prev=self.a_prev.clone(),ep_step=self.ep_step.clone())

    def _publish(self,transition=None):
        info={SUPERMIMIC_STATE_KEY:self.info()}
        if transition is not None:
            info[SUPERMIMIC_TRANSITION_KEY]=transition
        self.state=CollectorState(info)
        return self.state

    def init_state(self):
        return self._publish()

    def reset(self,ids,frames=None):
        if not ids.numel():
            return
        n=ids.numel()
        if frames is None:
            frames,bins=self.sample_frames(n)
        else:
            bins=self.motion.frame_bin[frames]
        p,R,v,w,q,qd=[x.clone() for x in self.motion.ref_state(frames)]
        nj=ROBOT.num_joints
        if self.randomization:
            p+=self.uniform(-1,1,(n,3))*torch.tensor([.05,.05,.01],device=self.device)
            R=so3_exp(self.uniform(-1,1,(n,3))*torch.tensor([.1,.1,.2],device=self.device))@R
            v+=self.uniform(-1,1,(n,3))*torch.tensor([.5,.5,.2],device=self.device)
            w+=self.uniform(-1,1,(n,3))*torch.tensor([.52,.52,.78],device=self.device)
            q+=self.uniform(-.1,.1,(n,nj))
            self.xi[ids,0]=self.uniform(.3,1.2,(n,))
            self.xi[ids,1:1+nj]=self.uniform(-.01,.01,(n,nj))
        q=q.clamp(self.kin.q_lo,self.kin.q_hi)
        qpos=torch.zeros(n,self.sim.model.nq,device=self.device)
        qvel=torch.zeros(n,self.sim.model.nv,device=self.device)
        qpos[:,:3]=p;qpos[:,3:7]=mat_to_quat(R);qpos[:,self.sim.qadr]=q
        qvel[:,:3]=v;qvel[:,3:6]=(R.transpose(-1,-2)@w[...,None]).squeeze(-1)
        qvel[:,self.sim.vadr]=qd
        self.sim.randomize(ids,self.xi[ids,0],self.xi[ids,1+nj:])
        self.sim.reset(ids,qpos,qvel)
        self.f[ids]=frames;self.ep_step[ids]=0;self.a_prev[ids]=0
        self.ep_ret[ids]=0;self.ep_disc[ids]=0;self.start_bin[ids]=bins

    def reset_to_frames(self,frames):
        frames=frames.to(self.device)
        if frames.shape!=(self.num_envs,) or frames.dtype!=torch.long:
            raise ValueError('reset frames must be int64 with one frame per environment')
        if bool(((frames<0)|(frames>=self.motion.num_frames)).any()):
            raise ValueError('reset frame outside library')
        self.reset(torch.arange(self.num_envs,device=self.device),frames)
        self._publish()
        return {},self.state.info

    @torch.no_grad()
    def step(self,action):
        if action.shape!=self.a_prev.shape or not bool(torch.isfinite(action).all()):
            raise ValueError('invalid action shape or non-finite actions')
        before=self.info()
        target=self.default_q+self.xi[:,1:1+ROBOT.num_joints]+self.alpha*action
        self.sim.step(target)
        next_state=self.read()
        end=self.motion.clip_end[self.motion.clip_id[self.f]]
        next_f=torch.minimum(self.f+1,end)
        reward=self.supermimic_mdp.reward(next_state,next_f,action,self.a_prev)
        hard=self.supermimic_mdp.termination(next_state,next_f)
        self.ep_ret+=reward;self.ep_disc+=self.gamma**self.ep_step.float()*reward
        self.ep_step+=1
        timeout=(next_f>=end)
        if not self.eval_mode:
            timeout|=self.ep_step>=self.max_episode_steps
        done=hard|timeout
        transition=dict(s=before['s'],s1=flatten_state(next_state).clone(),a=action.clone(),
            a_prev=before['a_prev'],xi=before['xi'],r=reward.clone(),f=before['f'],
            ep_step=before['ep_step'],term=hard.clone(),done=done.clone())
        if bool(done.any()):
            lengths=self.ep_step[done].float()
            normalized=self.ep_ret[done]/(lengths*self.supermimic_mdp.max_step_reward)
            transition.update(ep_len=lengths,ep_ret_norm=normalized)
            bootstrap=torch.where(hard[done],0.,self.gamma**lengths*self.ep_ret[done]/lengths/(1-self.gamma))
            returns=self.ep_disc[done]+bootstrap
            bins=self.start_bin[done]
            sums=torch.zeros_like(self.bin_G).index_add_(0,bins,returns)
            counts=torch.zeros_like(self.bin_G).index_add_(0,bins,torch.ones_like(returns))
            hit=counts>0
            self.bin_G[hit]=.8*self.bin_G[hit]+.2*sums[hit]/counts[hit]
        self.f=next_f;self.a_prev=action.clone()
        self.reset(done.nonzero().flatten())
        return self._publish(transition)

    def close(self):
        self.sim.close()
