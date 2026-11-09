#!/usr/bin/env python3
"""Hold out intermediate source phases without changing reference frame rate."""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil


def plan(manifest, every=3):
    if every < 2:
        raise ValueError('every must be at least two')
    primitives = {(c['name'].split('__')[0], c['name'].split('__primitive__')[1]): c for c in manifest['clips']
                  if '__primitive__' in c['name']}
    groups = {}
    for clip in manifest['clips']:
        match = re.search(r'([^_]+(?:_[^_]+)*)__to__([^_]+(?:_[^_]+)*)__a(\d+)__c(\d+)$', clip['name'])
        if not match:
            if '__primitive__' not in clip['name']:
                raise ValueError(f'Unrecognized clip: {clip["name"]}')
            continue
        source, target, stop, start = match.groups()
        group = clip['name'].split('__')[0]
        if int(start) != 0:
            raise ValueError('This experiment requires target entry mode')
        length = primitives[(group, source)]['source_frames']
        cutback = length-int(stop)
        if cutback < 0 or cutback % 15:
            raise ValueError('Expected source phases spaced by 15 frames at 30Hz')
        groups.setdefault((group, source, target), []).append((int(stop), cutback//15, clip))
    train = {c['name'] for c in primitives.values()}
    interpolation, extrapolation, rows = set(), set(), []
    for (group, source, target), clips in groups.items():
        selected = [stop for stop, tick, _ in clips if tick % every == 0]
        if not selected:
            raise ValueError(f'{source}->{target} has no sparse anchors')
        for stop, tick, clip in clips:
            name = clip['name']
            subset = 'train' if tick % every == 0 else (
                'interpolation' if min(selected) < stop < max(selected) else 'extrapolation')
            {'train': train, 'interpolation': interpolation, 'extrapolation': extrapolation}[subset].add(name)
            rows.append(dict(name=name, group=group, source=source, target=target, source_stop_frame_30hz=stop,
                             cutback_seconds=tick*.5, subset=subset))
    if train & (interpolation | extrapolation) or interpolation & extrapolation:
        raise AssertionError('Split overlap')
    return dict(train=train, interpolation=interpolation, extrapolation=extrapolation,
                heldout=interpolation | extrapolation, dense={c['name'] for c in manifest['clips']}), rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/k1'))
    p.add_argument('--groups', nargs='+', default=['boxing', 'traffic_control1', 'traffic_control2'])
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--every', type=int, default=3, help='Keep every Nth 0.5s cutback; default gives 1.5s')
    a = p.parse_args()
    if a.output_dir.exists():
        p.error('Use a new output directory')
    # Validate all groups before writing any output.
    plans = []
    for group in a.groups:
        source = a.data_root/group
        manifest = json.loads((source/'manifest.json').read_text())
        subsets, rows = plan(manifest, a.every)
        for c in manifest['clips']:
            if not (source/(c['name']+'.csv')).is_file():
                raise FileNotFoundError(c['name'])
        plans.append((group, source, manifest, subsets, rows))
    summary = {}
    for group, source, manifest, subsets, rows in plans:
        output = a.output_dir/group
        for subset, names in subsets.items():
            destination = output/subset
            destination.mkdir(parents=True)
            view = copy.deepcopy(manifest)
            view['clips'] = [c for c in view['clips'] if c['name'] in names]
            view['phase_split'] = dict(subset=subset, every=a.every,
                                       source_manifest=str((source/'manifest.json').resolve()))
            for c in view['clips']:
                target = destination/(c['name']+'.csv')
                shutil.copy2(source/target.name, target)
                c['csv'] = str(target.resolve())
            (destination/'manifest.json').write_text(json.dumps(view, indent=2)+'\n')
        primitive_count = sum('__primitive__' in c['name'] for c in manifest['clips'])
        counts = {key: len(value) for key, value in subsets.items()}
        summary[group] = dict(counts=counts, primitive_count=primitive_count,
                              train_transitions=counts['train']-primitive_count,
                              dense_transitions=counts['dense']-primitive_count)
        (output/'split.json').write_text(json.dumps(dict(summary=summary[group], phases=rows), indent=2)+'\n')
    (a.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
