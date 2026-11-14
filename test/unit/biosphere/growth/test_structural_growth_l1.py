"""
L1 Unit Tests for Structural Growth Parameter Integration

This test module verifies the canonical LAI-development morphism driven by the
explicit biomass-allocation growth port.

Test Coverage:
- Explicit leaf-growth and current-LAI input contracts
- LAI (Leaf Area Index) parameters
- Allometry parameters (future implementation)
- Basic compute_rate() functionality

References:
- Malhi et al. (2011). Productivity variation along elevation gradient. J Ecology 99:1-17
- Enquist (2002). Universal scaling in tree allometry. Nature 421:795-799
- Niklas & Enquist (2001). Invariant scaling relationships for interspecific plant biomass. Nature 410:655-660
"""

import pytest
import torch
import sys
import os
import inspect

# Add relic to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from relic.parameters import get_global_manager
from relic.core.functor import VariableKind
from relic.core.state import UpdateItem, UpdateMode
import relic.scientific.biosphere.growth as growth_package
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth


EXPECTED_OUTPUTS = {
    "diagnostics.structural_growth.gross_leaf_area_growth_tendency",
    "diagnostics.structural_growth.leaf_area_senescence_tendency",
    "diagnostics.structural_growth.net_leaf_area_tendency",
    "diagnostics.structural_growth.step_leaf_area_delta",
}


def output_values(output):
    """Return tensor values after asserting explicit update semantics."""
    assert set(output) == EXPECTED_OUTPUTS
    assert all(isinstance(item, UpdateItem) for item in output.values())
    return {key: item.value for key, item in output.items()}


@pytest.fixture(scope="module")
def param_manager():
    """Get global parameter manager instance."""
    return get_global_manager()


# =============================================================================
# Interface and parameter contracts
# =============================================================================


class TestStructuralGrowthContract:
    """Verify the diagnostic-only observer boundary."""

    def test_required_inputs_and_dependency_are_explicit(self):
        functor = StructuralGrowth(name="test_structural_growth")
        assert functor.get_required_inputs() == ["growth_rate_leaf", "leaf_area_index"]
        assert functor.get_dependencies() == ["biomass_allocation"]

    def test_metadata_declares_only_namespaced_diagnostics(self):
        functor = StructuralGrowth(name="test_structural_growth")
        assert set(functor.metadata.output_variables) == EXPECTED_OUTPUTS
        assert set(functor.get_provides()) == EXPECTED_OUTPUTS
        assert functor.metadata.variable_kind is VariableKind.DIAGNOSTIC
        assert "leaf_area_index" not in functor.metadata.output_variables
        assert "rate" not in functor.metadata.output_variables
        assert "sla" in functor.metadata.parameters
        assert "leaf_turnover_rate" in functor.metadata.parameters
        assert "senescence_rate" not in functor.metadata.parameters
        assert (
            functor.metadata.parameters["sla"]["namespace"]
            == "biosphere.phenology.morphology"
        )
        assert (
            functor.metadata.parameters["leaf_turnover_rate"]["namespace"]
            == "biosphere.phenology.turnover"
        )

    def test_package_exposes_no_compatibility_gate(self):
        assert not hasattr(growth_package, "_show_compatibility_warning")
        assert not hasattr(growth_package, "_allow_standard_execution")
        assert not hasattr(growth_package, "_COMPATIBILITY_WARNING_SHOWN")

    def test_rate_core_reads_the_state_only(self):
        """Item 114: the rate core, called by forward, takes the state only; the unused
        environment parameter (forward passed an empty mapping) is removed."""
        parameters = inspect.signature(StructuralGrowth._compute_rate_core).parameters
        assert list(parameters) == ["self", "state"]
        assert "_compute_rate_core(state)" in inspect.getsource(StructuralGrowth.forward)

    def test_reaction_map_is_signed_and_has_no_unit_conversion(self):
        forward_source = inspect.getsource(StructuralGrowth.forward)
        rate_source = inspect.getsource(StructuralGrowth._compute_rate_core)
        assert "torch.eye" not in forward_source
        assert "grams_to_kilograms" not in rate_source
        assert "SECONDS_PER_YEAR" not in rate_source

    @pytest.mark.parametrize(
        ("key", "units"),
        (
            ("biosphere.phenology.morphology.sla", "m2 kg-1 C"),
            ("biosphere.phenology.turnover.leaf_turnover_rate", "s-1"),
        ),
    )
    def test_parameter_units(self, param_manager, key, units):
        spec = param_manager.get_parameter(key)
        assert spec is not None
        assert spec.units == units


# =============================================================================
# Runtime and scientific numerics
# =============================================================================


class TestStructuralGrowthRuntime:
    """Verify numerical behavior and explicit UpdateItem semantics."""

    def test_fixed_numerics_and_update_contract(self, param_manager):
        functor = StructuralGrowth(name="test_structural_growth")
        leaf_growth = 2.0e-7
        leaf_area_index = 3.0
        dt = 3600.0
        output = functor.forward(
            {
                "growth_rate_leaf": torch.tensor(leaf_growth),
                "leaf_area_index": torch.tensor(leaf_area_index),
            },
            time=0.0,
            dt=dt,
        )
        values = output_values(output)

        sla = float(param_manager.get("biosphere.phenology.morphology.sla"))
        turnover = float(
            param_manager.get("biosphere.phenology.turnover.leaf_turnover_rate")
        )
        expected_growth = leaf_growth * sla
        expected_senescence = leaf_area_index * turnover
        expected_net = expected_growth - expected_senescence
        expected_step = expected_net * dt

        assert values[
            "diagnostics.structural_growth.gross_leaf_area_growth_tendency"
        ].item() == pytest.approx(expected_growth, rel=1.0e-6)
        assert values[
            "diagnostics.structural_growth.leaf_area_senescence_tendency"
        ].item() == pytest.approx(expected_senescence, rel=1.0e-6)
        assert values[
            "diagnostics.structural_growth.net_leaf_area_tendency"
        ].item() == pytest.approx(expected_net, rel=1.0e-6)
        assert values[
            "diagnostics.structural_growth.step_leaf_area_delta"
        ].item() == pytest.approx(expected_step, rel=1.0e-6)

        for key, item in output.items():
            assert item.mode is UpdateMode.REPLACE
            assert item.source_functor == "test_structural_growth"
            if key.endswith("step_leaf_area_delta"):
                assert item.unit == "m2 m-2"
                assert item.per_timestep is True
            else:
                assert item.unit == "m2 m-2 s-1"
                assert item.per_timestep is False

    def test_dt_changes_only_the_step_diagnostic(self):
        functor = StructuralGrowth(name="test_structural_growth")
        state = {
            "growth_rate_leaf": torch.tensor(2.0e-7),
            "leaf_area_index": torch.tensor(3.0),
        }
        short = output_values(functor.forward(state, time=0.0, dt=60.0))
        long = output_values(functor.forward(state, time=0.0, dt=3600.0))
        tendency_keys = EXPECTED_OUTPUTS - {
            "diagnostics.structural_growth.step_leaf_area_delta"
        }
        for key in tendency_keys:
            assert torch.equal(short[key], long[key])
        assert torch.allclose(
            long["diagnostics.structural_growth.step_leaf_area_delta"],
            short["diagnostics.structural_growth.step_leaf_area_delta"] * 60.0,
        )

    def test_zero_growth_has_direct_per_second_senescence(self, param_manager):
        functor = StructuralGrowth(name="test_structural_growth")
        leaf_area_index = 5.0
        output = output_values(
            functor.forward(
                {
                    "growth_rate_leaf": torch.tensor(0.0),
                    "leaf_area_index": torch.tensor(leaf_area_index),
                },
                time=0.0,
                dt=1.0,
            )
        )
        turnover = float(
            param_manager.get("biosphere.phenology.turnover.leaf_turnover_rate")
        )
        expected = leaf_area_index * turnover
        assert output[
            "diagnostics.structural_growth.leaf_area_senescence_tendency"
        ].item() == pytest.approx(expected, rel=1.0e-6)
        assert output[
            "diagnostics.structural_growth.net_leaf_area_tendency"
        ].item() == pytest.approx(-expected, rel=1.0e-6)

    def test_no_state_mutation_or_lai_writer(self):
        functor = StructuralGrowth(name="test_structural_growth")
        state = {
            "growth_rate_leaf": torch.tensor(2.0e-7),
            "leaf_area_index": torch.tensor(3.0),
        }
        before = {key: value.clone() for key, value in state.items()}
        output = functor.forward(state, time=0.0, dt=60.0)
        assert "leaf_area_index" not in output
        assert all(torch.equal(state[key], before[key]) for key in state)

    def test_inputs_remain_differentiable(self):
        functor = StructuralGrowth(name="test_structural_growth")
        leaf_growth = torch.tensor(2.0e-7, requires_grad=True)
        lai = torch.tensor(3.0, requires_grad=True)
        output = output_values(
            functor.forward(
                {"growth_rate_leaf": leaf_growth, "leaf_area_index": lai},
                time=0.0,
                dt=60.0,
            )
        )
        output["diagnostics.structural_growth.step_leaf_area_delta"].backward()
        assert leaf_growth.grad is not None and torch.isfinite(leaf_growth.grad)
        assert lai.grad is not None and torch.isfinite(lai.grad)


class TestStructuralGrowthValidation:
    """Verify fail-fast boundaries without aliases or defaults."""

    @pytest.mark.parametrize("missing", ("growth_rate_leaf", "leaf_area_index"))
    def test_missing_required_input_fails(self, missing):
        state = {
            "growth_rate_leaf": torch.tensor(2.0e-7),
            "leaf_area_index": torch.tensor(3.0),
        }
        state.pop(missing)
        with pytest.raises(KeyError, match=missing):
            StructuralGrowth().forward(state, time=0.0, dt=1.0)

    @pytest.mark.parametrize(
        ("key", "value"),
        (
            ("growth_rate_leaf", -1.0e-7),
            ("leaf_area_index", -1.0),
            ("growth_rate_leaf", float("nan")),
            ("leaf_area_index", float("inf")),
        ),
    )
    def test_invalid_inputs_fail(self, key, value):
        state = {
            "growth_rate_leaf": torch.tensor(2.0e-7),
            "leaf_area_index": torch.tensor(3.0),
        }
        state[key] = torch.tensor(value)
        with pytest.raises(ValueError, match=key):
            StructuralGrowth().forward(state, time=0.0, dt=1.0)

    @pytest.mark.parametrize("dt", (0.0, -1.0, float("nan")))
    def test_invalid_dt_fails(self, dt):
        with pytest.raises(ValueError, match="dt"):
            StructuralGrowth().forward(
                {
                    "growth_rate_leaf": torch.tensor(2.0e-7),
                    "leaf_area_index": torch.tensor(3.0),
                },
                time=0.0,
                dt=dt,
            )

    def test_npp_cannot_replace_leaf_growth_port(self):
        with pytest.raises(KeyError, match="growth_rate_leaf"):
            StructuralGrowth().forward(
                {
                    "net_primary_productivity": torch.tensor(2.0e-7),
                    "leaf_area_index": torch.tensor(3.0),
                },
                time=0.0,
                dt=1.0,
            )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

