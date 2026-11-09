#!/usr/bin/env python3
"""Import the audited V8 K1 bank, preserving original transition provenance."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from th_supermimic.data.rmai import convert_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    source = json.loads((a.bank / 'manifest.json').read_text())
    batch = json.loads((a.bank / 'robots/batch_manifest.json').read_text())
    assert batch['completed'] + batch['resumed'] == 75 and batch['failed'] == 0
    transitions = {x['id']: x for x in source['transitions']}
    expected = set(transitions) | {'primitive__'+x['name'] for x in source['primitives']}
    assert len(expected) == 75
    assert not a.output_dir.exists(), a.output_dir
    staged, records = [], []
    for name in sorted(expected):
        path = a.bank / 'robots' / name / 'k1/final/robot_motion.npz'
        csv, meta = convert_file(path, 'k1')
        rec = dict(name=name, kind='transition' if name in transitions else 'primitive', **meta)
        if name in transitions:
            tr = transitions[name]
            rec['transition'] = {key: tr[key] for key in ['source','target','cutback_seconds','source_phase_s','bridge_range','fps','seed']}
            rec['bridge_range_seconds'] = [f/tr['fps'] for f in tr['bridge_range']]
        staged.append((name,csv)); records.append(rec)
    a.output_dir.mkdir(parents=True)
    for name,csv in staged:
        target = a.output_dir/(name+'.csv')
        np.savetxt(target,csv,delimiter=',',fmt='%.10g')
    for rec in records: rec['csv'] = str((a.output_dir/(rec['name']+'.csv')).resolve())
    manifest = dict(schema_version=1,robot='k1',fps=50,groups=['ai_sapiens_v8'],
                    clips=records,source_bank=str(a.bank.resolve()),
                    note='3 retargeted V8 primitives + 72 full A-prefix/bridge/B transitions; one fresh unified policy and WM. No original CSV/ONNX warm start.')
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Imported 75 clips (3 primitives + 72 transitions) at 50Hz',flush=True)


if __name__=='__main__':main()
