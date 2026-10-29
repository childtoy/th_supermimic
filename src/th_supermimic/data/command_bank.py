"""Reference-bank selection and rigid planar alignment for inference."""
import re


def parse_bank(records):
    primitives = {r['name'].split('__primitive__')[1]: i for i, r in enumerate(records) if '__primitive__' in r['name']}
    edges = []
    for i, r in enumerate(records):
        m = re.search(r'([^_]+(?:_[^_]+)*)__to__([^_]+(?:_[^_]+)*)__a(\d+)__c(\d+)$', r['name'])
        if m is None:
            continue
        source, target, stop, start = m.groups()
        if int(start) or source not in primitives or target not in primitives:
            continue
        bridge = r['source_frames'] - int(stop) - records[primitives[target]]['source_frames']
        if bridge < 1:
            raise ValueError('Invalid corpus timing')
        edges.append(dict(index=i, source=source, target=target, stop=int(stop), bridge=bridge))
    return primitives, edges


class CommandBank:
    def __init__(self, env, records):
        self.env, self.records = env, records
        self.primitives, self.edges = parse_bank(records)
        if len(self.primitives) < 2 or not self.edges:
            raise ValueError('Missing primitive/pair reference bank')
        self.fields = ('root_p', 'root_R', 'root_v', 'root_w', 'body_p', 'body_R', 'body_v', 'body_w')
        self.original = {key: getattr(env.motion, key).clone() for key in self.fields}

    def restore(self):
        for key, value in self.original.items():
            getattr(self.env.motion, key).copy_(value)

    def switch(self, source, target, phase, current_frame):
        from th_supermimic.rl.algos.supermimic.geometry import yaw_mat
        lib = self.env.motion
        candidates = [e for e in self.edges if e['source'] == source and e['target'] == target]
        if not candidates:
            raise ValueError(f'No edge {source} -> {target}')
        edge = min(candidates, key=lambda e: abs((e['stop'] - 1) / 30 - phase))
        lo = int(lib.clip_start[edge['index']])
        hi = int(lib.clip_end[edge['index']]) + 1
        anchor = lo + round((edge['stop'] - 1) * 50 / 30)
        rotation = yaw_mat(lib.root_R[current_frame]) @ yaw_mat(self.original['root_R'][anchor]).T
        offset = lib.root_p[current_frame] - rotation @ self.original['root_p'][anchor]
        offset[2] = 0
        for key in self.fields:
            x = self.original[key][lo:hi]
            transformed = rotation @ x if key.endswith('_R') else (rotation @ x[..., None]).squeeze(-1)
            if key.endswith('_p'):
                transformed = transformed + offset
            getattr(lib, key)[lo:hi] = transformed
        target_start = lo + round((edge['stop'] + edge['bridge']) * 50 / 30)
        return anchor, target_start, dict(edge, name=self.records[edge['index']]['name'], phase_error_s=(edge['stop']-1)/30-phase)


def reference_timing(frame, target_start, clip_end, transition_start=None):
    """Display timing from actual loaded frame bounds, in simulation seconds."""
    duration = max(0, clip_end-target_start)/50
    elapsed = min(duration, max(0, frame-target_start)/50)
    transition_duration = max(0, target_start-transition_start)/50 if transition_start is not None else 0
    transition_left = max(0, target_start-frame)/50
    return dict(motion_duration_s=duration, motion_elapsed_s=elapsed,
                motion_left_s=max(0,duration-elapsed),
                motion_progress=elapsed/duration if duration else 1.0,
                transition_duration_s=transition_duration,
                transition_left_s=transition_left,
                transition_progress=1-min(1,transition_left/transition_duration) if transition_duration else 1.0,
                reference_left_s=max(0,clip_end-frame)/50)
