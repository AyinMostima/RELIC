import pytest
import torch

from relic.parameters import get_global_manager
from relic.scientific.pedosphere.api_contract import (
    create_output_summary,
    StandardPedospherePools,
)


def test_output_summary_vector_tensors() -> None:
    output = {
        "carbon_leaching_flux": torch.tensor([1.0, 3.0]),
        "delta_fast_som_c": torch.tensor([0.2, 0.4]),
        "diagnostics.richards_convergence": torch.tensor([2.0, 4.0]),
    }

    summary = create_output_summary(output, dt=10.0)

    assert summary["carbon_leaching_flux_total"] == pytest.approx(20.0)
    assert summary["carbon_leaching_flux_rate"] == pytest.approx(2.0)
    assert summary["delta_fast_som_c_change"] == pytest.approx(3.0)
    assert summary["diagnostics.richards_convergence"] == pytest.approx(3.0)


def test_standard_pedosphere_pools_bounds_from_parameters() -> None:
    manager = get_global_manager()
    if manager is None:
        pytest.skip("ParameterManager not available")

    spec = manager.get_parameter("pedosphere.carbon_pools.fast_som_c")
    if spec is not None and spec.bounds:
        assert StandardPedospherePools.FAST_CARBON.min_value == pytest.approx(float(spec.bounds[0]))
        assert StandardPedospherePools.FAST_CARBON.max_value == pytest.approx(float(spec.bounds[1]))

    nh4_min = manager.get(
        "pedosphere.numerical.min_concentration_nh4",
        units="g N m-2",
    )
    nh4_max = manager.get(
        "pedosphere.numerical.max_concentration_nh4",
        units="g N m-2",
    )

    if StandardPedospherePools.NH4.min_value is not None:
        assert StandardPedospherePools.NH4.min_value == pytest.approx(float(nh4_min))
    if StandardPedospherePools.NH4.max_value is not None:
        assert StandardPedospherePools.NH4.max_value == pytest.approx(float(nh4_max))


def test_microbial_pool_bounds_come_from_the_owner_process_parameters() -> None:
    """The microbial carbon pool is bounded by the microbial-enzyme owner parameters, not
    by an environment_defaults entry."""
    from relic.scientific.pedosphere import api_contract

    manager = get_global_manager()
    definition = api_contract.StandardPedospherePools._POOL_DEFINITIONS["MICROBIAL_CARBON"]
    assert "bounds_key" not in definition
    spec = StandardPedospherePools.MICROBIAL_CARBON
    minimum = manager.get("pedosphere.decomposition.microbial_enzyme.microbial.min_biomass")
    maximum = manager.get("pedosphere.decomposition.microbial_enzyme.microbial.max_biomass")
    assert spec.min_value == pytest.approx(float(minimum))
    assert spec.max_value == pytest.approx(float(maximum))


def test_parameter_lookup_errors_propagate() -> None:
    """A failing manager is an error, not an absent bound or an empty pool list."""
    from relic.scientific.pedosphere import api_contract

    class _FailingManager:
        def get_parameter(self, key):
            raise RuntimeError("manager unavailable for " + key)

        def get(self, key, **kwargs):
            raise RuntimeError("manager unavailable for " + key)

        def get_namespace_parameters(self, namespace):
            raise RuntimeError("manager unavailable for " + namespace)

    with pytest.raises(RuntimeError, match="unavailable"):
        api_contract._get_parameter_bounds("pedosphere.carbon_pools.fast_som_c", _FailingManager())
    with pytest.raises(RuntimeError, match="unavailable"):
        api_contract._get_parameter_value("pedosphere.numerical.min_concentration_nh4", _FailingManager())
    with pytest.raises(RuntimeError, match="unavailable"):
        api_contract._collect_parameter_pool_names("pedosphere.carbon_pools", "_c", _FailingManager())
