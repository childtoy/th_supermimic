import pytest
from th_supermimic.data.playback import for_new_references


def test_playback_replaces_only_reference_contract_without_mutating_checkpoint():
    source = dict(schema=2, robot='k1', joint_names=['a'], model_sha256='abc', motions=['train'])
    weights = object()
    checkpoint = dict(contract=source, pi=weights)
    target = dict(source, motions=['full_sequence'])
    view = for_new_references(checkpoint, target)
    assert view['pi'] is weights
    assert checkpoint['contract']['motions'] == ['train']
    assert view['contract'] == target
    for key, value in [('robot', 'g1'), ('model_sha256', 'other'), ('joint_names', ['b']), ('schema', 3)]:
        with pytest.raises(ValueError, match='contract mismatch'):
            for_new_references(checkpoint, dict(target, **{key: value}))
    with pytest.raises(ValueError, match='Missing'):
        for_new_references({}, target)
