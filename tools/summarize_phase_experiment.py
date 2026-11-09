#!/usr/bin/env python3
"""Collect completed phase evaluations without selecting by heldout score."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_root', type=Path)
    a = p.parse_args()
    rows = []
    for mode in ['sparse', 'dense']:
        for path in sorted((a.output_root/mode).glob('k1_*/heldout/*/summary.json')):
            run = path.parents[2].name
            checkpoint = path.parent.name
            for subset, metrics in json.loads(path.read_text()).items():
                rows.append(dict(mode=mode, group=run.removeprefix('k1_'), checkpoint=checkpoint,
                                 subset=subset, **metrics))
    if not rows:
        p.error('No completed phase evaluations found')
    (a.output_root/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    fields = ['mode', 'group', 'checkpoint', 'subset', 'n_clips', 'norm_return', 'n_failed', 'seen_training_clips', 'skipped']
    with (a.output_root/'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['mode']:6} {row['group']:16} {row['checkpoint']:13} {row['subset']:13} "
              f"return={row.get('norm_return', 0):.4f} failed={row.get('n_failed', 0)}/{row['n_clips']}")


if __name__ == '__main__':
    main()
