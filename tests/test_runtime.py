"""Optional end-to-end CPU backend test including actual learner updates."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from th_supermimic.robots import get_robot
from th_supermimic.data.rmai import convert_file

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('robot',['k1','g1'])
def test_simulator_world_model_actor_critic(tmp_path,robot):
    if not importlib.util.find_spec('mujoco'):
        pytest.skip('Install training runtime for end-to-end simulator test')
    if not (ROOT/'assets'/robot/'robot.xml').is_file():
        pytest.skip('Prepare robot assets first')
    spec=get_robot(robot)
    q=np.tile(np.r_[0,0,spec.root_height,1,0,0,0,spec.default_joint_pos],(80,1))
    npz=tmp_path/'fixture.npz';np.savez(npz,qpos=q,joint_names=spec.joint_names,fps=50,robot_id=robot)
    csv,metadata=convert_file(npz,robot)
    np.savetxt(tmp_path/'clip.csv',csv,delimiter=',')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(robot=robot,fps=50,clips=[dict(name='clip',**metadata)])))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=subprocess.run([sys.executable,str(ROOT/'tests/run_without_frameworks.py'),'--robot',robot,
        '--motion-dir',str(tmp_path),'--backend','mujoco','--run-dir',str(tmp_path/'run'),
        '--smoke'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert (tmp_path/'run/ckpt_000003.pt').is_file()
    assert '"completed": true' in result.stdout
    assert '"pi_gradnorm":' in result.stdout
