"""Replay must retain terminal states before row-selective automatic reset."""
import json
import numpy as np
import torch
from th_supermimic.robots import ROBOT
from th_supermimic.sim.env import TrackingEnv
from th_supermimic.rl.algos.supermimic.geometry import unflatten_state
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_partial_reset_preserves_other_rows_and_terminal_transition(tmp_path):
    assets=ROOT/'assets'/ROBOT.name
    if not (assets/'robot.xml').exists():
        pytest.skip('Prepare robot assets')
    row=np.r_[0,0,ROBOT.root_height,0,0,0,1,ROBOT.default_joint_pos]
    clip=np.tile(row,(20,1));clip[:,0]=np.arange(20)*.01
    path=tmp_path/'clip.csv';np.savetxt(path,clip,delimiter=',')
    env=TrackingEnv(assets,[str(path)],num_envs=2,backend='mujoco',randomization=False,max_episode_steps=100)
    try:
        env.reset_to_frames(torch.tensor([0,18]))
        prev=env.state.info['supermimic_state']
        result=env.step(torch.zeros(2,ROBOT.num_joints)).info
        t=result['supermimic'];c=result['supermimic_state']
        assert t['done'].tolist()==[False,True]
        assert c['ep_step'].tolist()==[1,0]
        assert c['f'][0]==1
        torch.testing.assert_close(t['s'],prev['s'])
        # Row zero is not reset; its next state must equal the collector state.
        torch.testing.assert_close(t['s1'][0],c['s'][0])
        # Terminal reference is at x ~ .19; reset is in an earlier reference bin.
        assert not torch.equal(t['s1'][1],c['s'][1])
        expected=env.supermimic_mdp.reward(unflatten_state(t['s1']),torch.tensor([1,19]),t['a'],t['a_prev'])
        torch.testing.assert_close(t['r'],expected)
        frozen=t['s1'].clone()
        env.step(torch.zeros(2,ROBOT.num_joints))
        torch.testing.assert_close(t['s1'],frozen)
    finally:
        env.close()
