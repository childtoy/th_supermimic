"""Explicit inference-only reference replacement with physical contract checks."""


def for_new_references(checkpoint, target_contract):
    source = checkpoint.get('contract')
    required = {'schema', 'robot', 'joint_names', 'model_sha256', 'motions'}
    if not isinstance(source, dict) or not required <= source.keys() or not required <= target_contract.keys():
        raise ValueError('Missing physical contract fields')
    physical = lambda contract: {k: v for k, v in contract.items() if k != 'motions'}
    if physical(source) != physical(target_contract):
        raise ValueError('Checkpoint robot/model contract mismatch for playback')
    # In-memory inference view only. Never rewrite source weights or checkpoint.
    return dict(checkpoint, contract=target_contract)
