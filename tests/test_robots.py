import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('robot', ['k1', 'g1'])
def test_fk_and_model_gradient(robot):
    if not (ROOT/'assets'/robot/'robot.xml').is_file():
        pytest.skip('run tools/prepare_assets.py first')
    env = dict(os.environ, TH_SUPERMIMIC_ROBOT=robot,
               TH_SUPERMIMIC_ASSETS=str(ROOT/'assets'/robot), PYTHONPATH=str(ROOT/'src'))
    subprocess.run([sys.executable,str(ROOT/'tests/check_robot.py')],env=env,check=True)
