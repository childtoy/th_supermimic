from types import SimpleNamespace
from pathlib import Path
import json
import torch
from th_supermimic.rl.algos.supermimic.learner import SuperMimicLearner


def test_best_return_and_coverage_are_separate_and_rng_is_preserved(tmp_path):
    saved=[]
    fake=SimpleNamespace(device=torch.device('cpu'),log_dir=tmp_path,iteration=1,evaluation_best={},
        _checkpoint=lambda **_: {},save=lambda path:saved.append(Path(path).name))
    state=torch.random.get_rng_state()
    def evaluate(_):
        torch.rand(100)
        return dict(norm_return=.8,n_failed=0)
    SuperMimicLearner._periodic_evaluation(fake,evaluate)
    assert torch.equal(torch.random.get_rng_state(),state)
    assert saved==['best_return.pt','best_coverage.pt']
    saved.clear();fake.iteration=2
    SuperMimicLearner._periodic_evaluation(fake,lambda _:dict(norm_return=.9,n_failed=3))
    assert saved==['best_return.pt']
    assert (tmp_path/'evals/eval_000002.json').is_file()
    assert fake.evaluation_best['coverage']==(0,.8)
