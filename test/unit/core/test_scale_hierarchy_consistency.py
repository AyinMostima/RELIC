
from relic.core.constants import SCALE_HIERARCHY
from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace


def test_scale_hierarchy_matches_parameter_database():
    """Hardcoded hierarchy must stay in sync with parameter database."""
    manager = get_global_manager()
    assert manager is not None
    levels = manager.get(
        "levels",
        namespace=resolve_namespace(ParameterNamespace.COUPLING_SCALES_HIERARCHY),
        expected_type=list,
        required=True,
    )
    assert SCALE_HIERARCHY == levels

