from types import SimpleNamespace
import torch
from th_supermimic.rl.algos.supermimic.learner import SuperMimicLearner


def test_bootstrap_upper_bound_respects_maximum_possible_discounted_reward():
    values=torch.tensor([-4.,8.,69.],requires_grad=True)
    fake=SimpleNamespace(critic=lambda _:values,
        cfg=SimpleNamespace(critic_upper_bound=True,gamma=.99),
        mdp=SimpleNamespace(max_step_reward=.1))
    bounded=SuperMimicLearner._bootstrap_value(fake,None)
    torch.testing.assert_close(bounded,torch.tensor([-4.,8.,10.]))
    bounded.sum().backward()
    torch.testing.assert_close(values.grad,torch.tensor([1.,1.,0.]))
    # The raw prediction remains available for critic regression to a valid target.
    values.grad.zero_();((values-bounded.detach())**2).sum().backward()
    assert values.grad[-1]>0


def test_default_bootstrap_is_unchanged_for_control_experiments():
    values=torch.tensor([69.],requires_grad=True)
    fake=SimpleNamespace(critic=lambda _:values,cfg=SimpleNamespace(critic_upper_bound=False))
    assert SuperMimicLearner._bootstrap_value(fake,None) is values
