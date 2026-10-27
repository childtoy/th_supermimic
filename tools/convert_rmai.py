#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from th_supermimic.data.rmai import convert_corpus

p = argparse.ArgumentParser(description='Import only the chosen robot from a RMAI retarget corpus')
p.add_argument('--input-dir', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--robot', choices=['k1', 'g1'], required=True)
p.add_argument('--groups', nargs='+', choices=['boxing', 'traffic_control1', 'traffic_control2'],
               default=['boxing', 'traffic_control1', 'traffic_control2'])
if __name__ == '__main__':
    a = p.parse_args()
    m = convert_corpus(a.input_dir, a.output_dir, a.robot, a.groups)
    print(f"Imported {len(m['clips'])} {a.robot} clips at 50 Hz: {a.output_dir / 'manifest.json'}")
