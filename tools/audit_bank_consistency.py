#!/usr/bin/env python3
"""Compare pair source prefixes and target suffixes against canonical primitives.

Suffixes are aligned at their last resampled frame. This may leave a sub-frame
timing offset and is a diagnostic, not proof of the upstream cause.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


def audit(directory):
    records = json.loads((directory / 'manifest.json').read_text())['clips']
    primitives = {r['name'].split('__primitive__')[1]: r for r in records if '__primitive__' in r['name']}
    cache = {}

    def read(record):
        name = record['name']
        if name not in cache:
            with (directory / (name + '.csv')).open() as stream:
                cache[name] = [[float(x) for x in row] for row in csv.reader(stream)]
        return cache[name]

    def rmse(first, second):
        joints = len(first[0])-7
        errors = [math.sqrt(statistics.mean((a[j+7]-b[j+7])**2 for a, b in zip(first, second)))
                  for j in range(joints)]
        return math.sqrt(statistics.mean(x*x for x in errors)), errors

    rows = []
    for record in records:
        match = re.search(r'([^_]+(?:_[^_]+)*)__to__([^_]+(?:_[^_]+)*)__a(\d+)__c(\d+)$', record['name'])
        if not match:
            continue
        source, target, stop, start = match.groups()
        if int(start):
            continue
        pair, a, b = read(record), read(primitives[source]), read(primitives[target])
        n = min(round((int(stop)-1)*50/30)+1, len(pair), len(a))
        source_error, _ = rmse(pair[:n], a[:n])
        n = min(len(pair), len(b))
        target_error, per_joint = rmse(pair[-n:], b[-n:])
        names = record['joint_names']
        rows.append(dict(name=record['name'], source=source, target=target,
                         source_prefix_rmse_rad=source_error, target_suffix_rmse_rad=target_error,
                         target_per_joint_rmse_rad=dict(zip(names, per_joint))))
    return dict(pair_count=len(rows),
                mean_source_prefix_rmse_rad=statistics.mean(r['source_prefix_rmse_rad'] for r in rows),
                mean_target_suffix_rmse_rad=statistics.mean(r['target_suffix_rmse_rad'] for r in rows),
                target_suffix_over_0_5_rad=sum(r['target_suffix_rmse_rad']>.5 for r in rows),
                per_clip=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/k1'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {group: audit(a.data_root / group) for group in ['boxing', 'traffic_control1', 'traffic_control2']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({g: {k: v for k, v in r.items() if k != 'per_clip'} for g, r in result.items()}, indent=2))


if __name__ == '__main__':
    main()
