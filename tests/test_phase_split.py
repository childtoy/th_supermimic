"""Leakage and namespace regressions for the sparse-phase experiment."""
from pathlib import Path
import runpy
import unittest

plan = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/split_transition_phases.py'))['plan']


def clips(group, length):
    return [dict(name=f'{group}__primitive__{name}', source_frames=length) for name in ['a', 'b']] + [
        dict(name=f'{group}__{group}__a__to__b__a{stop:05d}__c00000')
        for stop in range(length, 9, -15)]


class PhaseSplitTest(unittest.TestCase):
    def test_interpolation_and_extrapolation_are_separate(self):
        subsets, rows = plan(dict(clips=clips('one', 90)))
        stops = lambda subset: {r['source_stop_frame_30hz'] for r in rows if r['subset']==subset}
        self.assertEqual(stops('train'), {90, 45})
        self.assertEqual(stops('interpolation'), {75, 60})
        self.assertEqual(stops('extrapolation'), {30, 15})
        self.assertFalse(subsets['train'] & subsets['heldout'])

    def test_all_keeps_identical_action_names_in_separate_groups(self):
        first, _ = plan(dict(clips=clips('one', 90)))
        second, _ = plan(dict(clips=clips('two', 75)))
        combined, _ = plan(dict(clips=clips('one', 90)+clips('two', 75)))
        for subset in combined:
            self.assertEqual(combined[subset], first[subset] | second[subset])

    def test_short_source_does_not_claim_interpolation(self):
        subsets, _ = plan(dict(clips=clips('short', 40)))
        self.assertFalse(subsets['interpolation'])
        self.assertTrue(subsets['extrapolation'])


if __name__ == '__main__':
    unittest.main()
