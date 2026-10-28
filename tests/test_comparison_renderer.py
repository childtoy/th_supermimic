from pathlib import Path
import pytest
from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import load_comparison_model

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('robot,nq',[('k1',30),('g1',36)])
def test_portable_model_can_render_reference_ghost(robot,nq):
    scene=ROOT/'assets'/robot/'scene.xml'
    if not scene.is_file():pytest.skip('Prepare robot assets')
    m=load_comparison_model(scene)
    assert m.nq==2*nq
    assert m.body('ref_pelvis').id>m.body('pelvis').id
    for geom in range(m.ngeom):
        if m.geom(geom).name.startswith('ref_'):
            assert m.geom_contype[geom]==0 and m.geom_conaffinity[geom]==0
