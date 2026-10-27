#!/usr/bin/env python3
"""Build portable position-controlled MJCFs from the exact RIMKit robot assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from th_supermimic.robots import get_robot


def prepare(robot, rimkit_root, output):
    spec = get_robot(robot)
    robot_assets = rimkit_root / 'src/rimkit/assets/robots'
    relative = 'robotis/k1/k1.xml' if robot == 'k1' else 'unitree/g1/g1_29dof.xml'
    source = robot_assets / relative
    tree = ET.parse(source)
    root = tree.getroot()
    output.mkdir(parents=True, exist_ok=True)
    compiler = root.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(root, 'compiler')
    meshdir = source.parent / compiler.get('meshdir', '')
    # Copy exact mesh files and flatten file references. These assets are generated,
    # not dependencies on an absolute workstation path.
    (output / 'meshes').mkdir(exist_ok=True)
    for mesh in root.findall('./asset/mesh'):
        origin = meshdir / mesh.get('file')
        target = output / 'meshes' / origin.name
        shutil.copy2(origin, target)
        mesh.set('file', origin.name)
    compiler.set('meshdir', 'meshes')
    compiler.set('fusestatic', 'false')
    joints = {j.get('name'): j for j in root.iter('joint') if j.get('name')}
    if not set(spec.joint_names) <= joints.keys():
        raise ValueError('MJCF joint set does not match selected robot')
    for n, motor in zip(spec.joint_names, spec.motors):
        joints[n].set('armature', str(motor['armature']))
        joints[n].set('damping', '0')
        joints[n].set('frictionloss', '0')
        joints[n].set('actuatorfrcrange', f"{-motor['effort']} {motor['effort']}")
    old = root.find('actuator')
    if old is not None:
        root.remove(old)
    actuators = ET.SubElement(root, 'actuator')
    for n, m in zip(spec.joint_names, spec.motors):
        ET.SubElement(actuators, 'position', name=n, joint=n, kp=str(m['kp']), kv=str(m['kd']),
                      ctrllimited='false', forcelimited='true',
                      forcerange=f"{-m['effort']} {m['effort']}")
    collision_names = []
    for body in root.iter('body'):
        for i, geom in enumerate(body.findall('geom')):
            # Name every robot geom so backend identity binding is deterministic.
            name = f"{body.get('name')}_geom_{i}"
            geom.set('name', name)
            if geom.get('contype', '1') != '0' or geom.get('conaffinity', '1') != '0':
                geom.set('contype', '1')
                geom.set('conaffinity', '0')
                collision_names.append(name)
    for element in list(root.findall('keyframe')):
        root.remove(element)
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', '0.005')
    option.set('integrator', 'implicitfast')
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 0.05',
                  contype='0', conaffinity='1', friction='0.75 0.005 0.0001', rgba='0.6 0.65 0.7 1')
    # MuJoCo determines qpos order from tree traversal, independent of action order.
    joint_order = [j.get('name') for j in root.iter('joint') if j.get('name') in spec.joint_names]
    q = dict(zip(spec.joint_names, spec.default_joint_pos))
    qpos = [0, 0, spec.root_height, 1, 0, 0, 0] + [q[n] for n in joint_order]
    ET.SubElement(ET.SubElement(root, 'keyframe'), 'key', name='stand',
                  qpos=' '.join(map(str, qpos)), ctrl=' '.join(map(str, spec.default_joint_pos)))
    ET.indent(root)
    # Both filenames include the same model to avoid include-path ambiguity.
    for name in ['robot.xml', 'scene.xml']:
        tree.write(output / name, encoding='unicode')
    metadata = dict(robot=robot, source_relative=relative,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    collision_geom_names=collision_names, joint_names=list(spec.joint_names),
                    tracked_bodies=list(spec.tracked_bodies))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # Preserve vendor attribution alongside generated copies.
    vendor = robot_assets / ('robotis' if robot == 'k1' else 'unitree')
    for name in ['LICENSE', 'SOURCE.yaml', 'MODIFICATIONS.md']:
        if (vendor / name).is_file():
            shutil.copy2(vendor / name, output / name)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str((output / 'robot.xml').resolve()))
    if model.nq != 7 + spec.num_joints or model.nu != spec.num_joints:
        raise ValueError(f'Unexpected model dimensions {model.nq=} {model.nu=}')
    for body in spec.tracked_bodies:
        model.body(body)
    print(f'{robot}: nq={model.nq} nv={model.nv} nu={model.nu}, assets={output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', choices=['k1', 'g1', 'all'], default='all')
    p.add_argument('--rimkit-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'assets')
    args = p.parse_args()
    for robot in (['k1', 'g1'] if args.robot == 'all' else [args.robot]):
        prepare(robot, args.rimkit_root.resolve(), (args.output_dir / robot).resolve())
