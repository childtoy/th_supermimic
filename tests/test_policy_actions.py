"""A policy must request the same bounded actions in collection, imagination and eval."""
import torch
from th_supermimic.rl.algos.supermimic.networks import Policy


def test_policy_clamps_extreme_actions_without_changing_checkpoint_shape():
    pi=Policy(4,5,hist=2,z_dim=3,num_joints=3,hidden=(8,),action_limit=20)
    with torch.no_grad():
        pi.dec[-1].weight.zero_()
        pi.dec[-1].bias.copy_(torch.tensor([-100.,2.,100.]))
    action=pi(torch.zeros(2,2,4),torch.zeros(2,5))
    torch.testing.assert_close(action,torch.tensor([[-20.,2.,20.],[-20.,2.,20.]]))
    action.sum().backward()
    # The in-range output retains policy gradients; saturated outputs are bounded.
    torch.testing.assert_close(pi.dec[-1].bias.grad,torch.tensor([0.,2.,0.]))
    restored=Policy(4,5,hist=2,z_dim=3,num_joints=3,hidden=(8,))
    restored.load_state_dict(pi.state_dict())
    torch.testing.assert_close(restored(torch.zeros(2,2,4),torch.zeros(2,5)),action)
