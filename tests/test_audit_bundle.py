"""An audit bundle must preserve exact checkpoint and reference bytes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import pytest

spec=importlib.util.spec_from_file_location('collect_audit',Path(__file__).resolve().parents[1]/'tools/collect_training_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture(root):
    for d in ['runs/k1_boxing','data/k1/boxing','assets/k1','src/th_supermimic','tools','scripts']:
        (root/d).mkdir(parents=True)
    (root/'runs/k1_boxing/last.pt').write_bytes(b'exact checkpoint bytes')
    (root/'runs/k1_boxing/ckpt_002000.pt').write_bytes(b'early checkpoint')
    (root/'runs/k1_boxing/run_config.json').write_text('{}')
    (root/'data/k1/boxing/manifest.json').write_text(json.dumps(dict(robot='k1',fps=50,clips=[dict(name='motion')])))
    (root/'data/k1/boxing/motion.csv').write_bytes(b'1,2,3\n')
    (root/'assets/k1/robot.xml').write_text('<mujoco/>')
    (root/'src/th_supermimic/data').mkdir()
    (root/'src/th_supermimic/data/rmai.py').write_text('# loader\n')


def test_bundle_is_complete_and_hashes_match(tmp_path):
    fixture(tmp_path)
    output=tmp_path/'exports/audit.tar.gz'
    audit.build_archive(tmp_path,output,['boxing'],[2000,4000])
    with tarfile.open(output) as tar:
        report=json.load(tar.extractfile('training_audit/audit_manifest.json'))
        assert report['missing_optional']==['runs/k1_boxing/ckpt_004000.pt']
        for row in report['files']:
            content=tar.extractfile('training_audit/'+row['path']).read()
            assert hashlib.sha256(content).hexdigest()==row['sha256']
            assert content==(tmp_path/row['path']).read_bytes()
        assert 'training_audit/src/th_supermimic/data/rmai.py' in tar.getnames()


def test_missing_training_reference_fails_before_archive(tmp_path):
    fixture(tmp_path)
    (tmp_path/'data/k1/boxing/motion.csv').unlink()
    with pytest.raises(FileNotFoundError):
        audit.build_archive(tmp_path,tmp_path/'exports/audit.tar.gz',['boxing'],[2000])
    assert not (tmp_path/'exports/audit.tar.gz').exists()
