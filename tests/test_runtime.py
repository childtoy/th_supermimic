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

    warm=subprocess.run([sys.executable,str(ROOT/'tests/run_without_frameworks.py'),'--robot',robot,
        '--motion-dir',str(tmp_path),'--backend','mujoco','--run-dir',str(tmp_path/'warm'),
        '--warm-start',str(tmp_path/'run/ckpt_000003.pt'),'--smoke','--eval-every','1',
        'algo.critic_upper_bound=true','algo.policy_lr_scale=0.25'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert warm.returncode==0,warm.stdout+'\n'+warm.stderr
    assert (tmp_path/'warm/ckpt_000003.pt').is_file()
    snapshot=json.loads((tmp_path/'warm/run_config.json').read_text())
    assert snapshot['warm_start']==str((tmp_path/'run/ckpt_000003.pt').resolve())
    assert snapshot['learner']['critic_upper_bound'] is True

    assert (tmp_path/'warm/best_return.pt').is_file()
    assert (tmp_path/'warm/best_coverage.pt').is_file()
    assert len(list((tmp_path/'warm/evals').glob('eval_*.json')))==4

    import torch
    full=torch.load(tmp_path/'warm/last.pt',map_location='cpu',weights_only=True)
    assert full['opt_pi']['param_groups'][0]['lr']==pytest.approx(full['opt_wm']['param_groups'][0]['lr']*.25)

    resumed=subprocess.run([sys.executable,str(ROOT/'tests/run_without_frameworks.py'),'--robot',robot,
        '--motion-dir',str(tmp_path),'--backend','mujoco','--run-dir',str(tmp_path/'resumed'),
        '--resume',str(tmp_path/'warm/last.pt'),'--num-envs','8','--iterations','4',
        '--eval-every','1'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert resumed.returncode==0,resumed.stdout+'\n'+resumed.stderr
    assert (tmp_path/'resumed/train_log.csv').read_text().startswith('iteration,')
    assert (tmp_path/'resumed/best_return.pt').is_file()
    assert (tmp_path/'resumed/evals/eval_000003.json').is_file()
    assert (tmp_path/'resumed/ckpt_000004.pt').is_file()
