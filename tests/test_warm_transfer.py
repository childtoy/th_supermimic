from types import SimpleNamespace
import pytest
import torch
from th_supermimic.rl.algos.supermimic.learner import SuperMimicLearner


def test_transfer_requires_explicit_permission_and_preserves_robot_contract(tmp_path):
    source=dict(schema=2,robot='k1',model_sha256='model-a',joint_names=['joint'],motions=['old'])
    wm=torch.nn.Linear(1,1);pi=torch.nn.Linear(1,1)
    with torch.no_grad():wm.weight.fill_(2);pi.weight.fill_(3)
    path=tmp_path/'source.pt'
    torch.save(dict(contract=source,wm=wm.state_dict(),pi=pi.state_dict(),iteration=2000),path)
    target=dict(source,motions=['new'])
    learner=SimpleNamespace(device='cpu',env=SimpleNamespace(contract=target),
                            wm=torch.nn.Linear(1,1),pi=torch.nn.Linear(1,1),iteration=0)
    with pytest.raises(ValueError,match='contract mismatch'):
        SuperMimicLearner.warm_start(learner,path)
    SuperMimicLearner.warm_start(learner,path,allow_motion_transfer=True)
    torch.testing.assert_close(learner.wm.weight,wm.weight)
    torch.testing.assert_close(learner.pi.weight,pi.weight)
    assert learner.iteration==0
    for key,value in [('robot','g1'),('model_sha256','another-model'),('joint_names',['other'])]:
        learner.env.contract=dict(target,**{key:value})
        with pytest.raises(ValueError,match='contract mismatch'):
            SuperMimicLearner.warm_start(learner,path,allow_motion_transfer=True)
