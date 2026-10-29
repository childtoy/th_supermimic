from types import SimpleNamespace
import torch
from th_supermimic.data.command_bank import CommandBank, parse_bank


def test_pair_name_and_timing():
    records=[dict(name='traffic_control2__primitive__firm_stop',source_frames=86),dict(name='traffic_control2__primitive__come_forward',source_frames=72),dict(name='traffic_control2__traffic_control2__firm_stop__to__come_forward__a00056__c00000',source_frames=173)]
    primitives,edges=parse_bank(records)
    assert primitives=={'firm_stop':0,'come_forward':1}
    assert edges==[dict(index=2,source='firm_stop',target='come_forward',stop=56,bridge=45)]


def test_bank_switch_preserves_physical_state_and_rigid_geometry():
    records=[dict(name='x__primitive__a',source_frames=60),dict(name='x__primitive__b',source_frames=60),dict(name='x__x__a__to__b__a00030__c00000',source_frames=135)]
    n=400
    root_R=torch.eye(3).repeat(n,1,1)
    root_R[0]=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])
    lib=SimpleNamespace(clip_start=torch.tensor([0,100,200]),clip_end=torch.tensor([99,199,399]),root_p=torch.randn(n,3),root_R=root_R,root_v=torch.randn(n,3),root_w=torch.randn(n,3),body_p=torch.randn(n,2,3),body_R=torch.eye(3).repeat(n,2,1,1),body_v=torch.randn(n,2,3),body_w=torch.randn(n,2,3))
    physical_state=torch.randn(30)
    env=SimpleNamespace(motion=lib,physical_state=physical_state.clone())
    bank=CommandBank(env,records)
    old_root=lib.root_p[0].clone()
    anchor,target_start,edge=bank.switch('a','b',1.0,0)
    torch.testing.assert_close(lib.root_p[anchor,:2],old_root[:2])
    torch.testing.assert_close(lib.root_R[anchor],lib.root_R[0])
    torch.testing.assert_close(lib.root_p[200:,2],bank.original['root_p'][200:,2])
    torch.testing.assert_close(env.physical_state,physical_state)
    torch.testing.assert_close(lib.body_v[200:].norm(dim=-1),bank.original['body_v'][200:].norm(dim=-1))
    assert target_start>anchor
    bank.restore()
    torch.testing.assert_close(lib.root_p,bank.original['root_p'])


def test_display_timing_at_start_transition_and_end():
    from th_supermimic.data.command_bank import reference_timing
    initial=reference_timing(100,100,200)
    assert initial['motion_elapsed_s']==0 and initial['motion_duration_s']==2
    midway=reference_timing(125,100,200)
    assert midway['motion_left_s']==1.5 and midway['motion_progress']==.25
    bridge=reference_timing(75,100,200,50)
    assert bridge['transition_left_s']==.5 and bridge['transition_progress']==.5
    assert bridge['motion_elapsed_s']==0
    final=reference_timing(200,100,200,50)
    assert final['motion_left_s']==0 and final['motion_progress']==1
