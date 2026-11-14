import torch

from relic.parameters import get_global_manager
from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
import pytest


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def test_first_order_decomposition_yaml_parameters():
    manager = get_global_manager(preload_all=True, warmup_cache=False)

    spec = manager.get_parameter("pedosphere.decomposition.k_metabolic")
    assert spec is not None
    assert spec.units == "yr-1"

    response_spec = manager.get_parameter(
        "pedosphere.decomposition.temp_response_type"
    )
    assert response_spec is not None
    assert response_spec.allowed_values == ["q10", "arrhenius"]
    assert manager.get_parameter("pedosphere.decomposition.litter_inputs") is None

    decomposer = FirstOrderDecomposition(name="yaml_decomposition", config={})

    k_metabolic = manager.get(
        "pedosphere.decomposition.k_metabolic",
        units="yr-1",
    )
    k_ref = decomposer.pool_params["metabolic_litter"]["k_ref"]
    expected = _as_float(k_metabolic) / _as_float(decomposer.seconds_per_year)

    assert abs(_as_float(k_ref) - expected) < 1e-12
