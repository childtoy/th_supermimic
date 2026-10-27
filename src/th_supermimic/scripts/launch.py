"""Select the robot before importing robot-dependent environment modules."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--robot', choices=['k1', 'g1'], required=True)
    p.add_argument('--motion-dir', type=Path, required=True)
    p.add_argument('--assets-dir', type=Path)
    p.add_argument('--backend', choices=['mjwarp', 'mujoco', 'motrix'], default='mjwarp')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--config-only', action='store_true', help='Validate and print resolved config without loading simulator')
    a, overrides = p.parse_known_args()
    assets = (a.assets_dir or Path('assets') / a.robot).resolve()
    os.environ['TH_SUPERMIMIC_ROBOT'] = a.robot
    os.environ['TH_SUPERMIMIC_ASSETS'] = str(assets)
    manifest = json.loads((a.motion_dir / 'manifest.json').read_text())
    if manifest['robot'] != a.robot or manifest['fps'] != 50:
        raise ValueError('Imported motion manifest robot/fps mismatch')
    files = [(a.motion_dir / (clip['name'] + '.csv')).resolve() for clip in manifest['clips']]
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError('Missing CSV motion files; run tools/convert_rmai.py')
    metadata = json.loads((assets / 'metadata.json').read_text())
    if metadata['robot'] != a.robot:
        raise ValueError('Asset robot mismatch')
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    config_dir = Path(__file__).resolve().parents[1] / 'conf/supermimic'
    requested = [f'task=humanoid_super_mimic/{a.backend}', f'robot={a.robot}']
    if a.smoke:
        requested += ['algo.num_envs=8', 'algo.iterations=3', 'algo.buffer_steps=64',
                      'algo.warmup_steps=16', 'algo.wm_only_iters=0', 'algo.wm_window=2',
                      'algo.horizon=4', 'algo.wm_batches=1', 'algo.wm_batch=8', 'algo.pi_batch=8',
                      'algo.wm_hidden=[64,64]', 'algo.policy_hidden=[64,64]',
                      'algo.critic_hidden=[64,32]', 'algo.critic_epochs=1',
                      'algo.critic_minibatches=1', 'algo.compile=false', 'algo.amp=none',
                      'algo.ckpt_every=1', 'algo.log_every=1']
    with initialize_config_dir(version_base='1.3', config_dir=str(config_dir)):
        cfg = compose(config_name='config', overrides=requested + overrides)
    # Robot selection and model paths cannot be changed through Hydra overrides.
    expected_joints = metadata['joint_names']
    if list(cfg.env.scene.entities.robot.joint_names) != expected_joints:
        raise ValueError('Config joints do not match selected assets')
    cfg.env.commands.motion.motion_files = [str(path) for path in files]
    cfg.env.commands.motion.src_fps = 50.0
    cfg.training.log_dir = str(a.run_dir.resolve())
    if a.config_only:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    from th_supermimic.scripts.train_supermimic import main as train
    train(cfg)


if __name__ == '__main__':
    main()
