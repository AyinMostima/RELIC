import pytest

from relic.coupling.scales.scale_manager import ScaleManager
from relic.parameters import get_global_manager


def _make_manager():
    manager = ScaleManager.__new__(ScaleManager)
    manager.parameter_manager = get_global_manager()
    manager._facade_warnings = []
    return manager


def test_unknown_variable_requires_metadata():
    manager = _make_manager()
    with pytest.raises(KeyError):
        manager.get_variable_metadata("custom_variable", scale="daily")


def test_undeclared_variables_raise_one_error_type_without_a_name_heuristic():
    """Flux-looking names get the same KeyError as any other undeclared variable."""
    import relic.coupling.scales.scale_manager as scale_manager_module

    manager = _make_manager()
    raised = []
    for name in ("custom_variable", "custom_flux", "evapotranspiration_rate_custom"):
        with pytest.raises(KeyError, match=name) as excinfo:
            manager.get_variable_metadata(name, scale="daily")
        raised.append(type(excinfo.value))
    assert raised == [KeyError, KeyError, KeyError]
    assert not hasattr(scale_manager_module, "FluxMetadataMissingError")
