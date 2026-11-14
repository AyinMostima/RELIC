"""L1 contract tests for the canonical land-use CTMC process."""

import warnings
from unittest.mock import patch

import pytest
import torch

from relic.core.abstractions import ScaleCoupling
from relic.core.functor import ScaleCouplingFunctor
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.anthroposphere.land_use.transitions import (
    LandUseTransitions,
    LandUseType,
    TransitionEvent,
)


CANONICAL_INPUTS = {
    "land_use_fraction",
    "time_index",
    "gdp_per_capita",
    "population_density",
    "market_accessibility",
    "policy_intensity",
}
CANONICAL_OUTPUT_UNITS = {
    "land_use_fraction": "1",
    "diagnostics.land_use_transitions.delta_fraction": "1",
    "diagnostics.land_use_transitions.dominant_type_index": "1",
    "diagnostics.land_use_transitions.maximum_fraction": "1",
    "diagnostics.land_use_transitions.soil_carbon_input_rate_change": (
        "kg C m-2 s-1"
    ),
    "diagnostics.land_use_transitions.canonical_iphs_runtime": "1",
}


def _normalized(shape: tuple[int, ...], type_axis: int) -> torch.Tensor:
    values = torch.rand(shape, dtype=torch.float64) + 0.1
    return values / values.sum(dim=type_axis, keepdim=True)


class TestLandUseTransitionsL1:
    """Verify the single canonical full-state transition path."""

    @pytest.fixture
    def functor(self) -> LandUseTransitions:
        return LandUseTransitions()

    @pytest.fixture
    def sample_state(self) -> dict[str, torch.Tensor]:
        return {
            "land_use_fraction": _normalized((2, 9, 4, 5), type_axis=1),
            "time_index": torch.tensor(3.0),
            "gdp_per_capita": torch.full((2, 4, 5), 25000.0),
            "population_density": torch.full((2, 4, 5), 250.0),
            "market_accessibility": torch.full((2, 4, 5), 0.4),
            "policy_intensity": torch.full((2, 4, 5), 0.2),
        }

    def test_class_instantiation(self, functor):
        assert functor.name == "land_use_transitions"
        assert functor.n_types == len(LandUseType) == 9
        assert functor.runtime_path == "canonical_iphs_land_use_ctmc_chain_map"

    def test_metadata_declares_exact_interface(self, functor):
        assert set(functor.metadata.input_variables) == CANONICAL_INPUTS
        assert set(functor.metadata.output_variables) == set(CANONICAL_OUTPUT_UNITS)
        assert set(functor.OUTPUT_UNITS) == set(CANONICAL_OUTPUT_UNITS)

    def test_required_parameters_are_canonical_namespaces(self, functor):
        assert set(functor.required_parameters()) == {
            "anthroposphere.land_use.transition_rates.*",
            "anthroposphere.land_use.transition_channels.*",
            "anthroposphere.land_use.numerical.*",
            "anthroposphere.land_use.soil.*",
            "validation.conservation.conservation_tolerance",
        }
        assert not any(
            key.startswith("relic.physical.constants")
            for key in functor.required_parameters()
        )

    @pytest.mark.parametrize(
        ("shape", "type_axis"),
        [
            ((9,), 0),
            ((3, 9), 1),
            ((9, 2, 4), 0),
            ((3, 9, 4), 1),
            ((3, 9, 2, 4), 1),
        ],
    )
    def test_supported_fraction_layouts(self, functor, shape, type_axis):
        fractions = _normalized(shape, type_axis)
        actual, actual_axis = functor._validate_fraction_state(
            {"land_use_fraction": fractions}
        )
        assert actual is fractions
        assert actual_axis == type_axis

    def test_missing_canonical_state_and_old_alias_fail_fast(self, functor):
        with pytest.raises(KeyError, match="land_use_fraction"):
            functor.forward({}, time=0.0, dt=1.0)
        with pytest.raises(KeyError, match="land_use_fraction"):
            functor.forward(
                {"land_use_fractions": torch.full((9,), 1.0 / 9.0)},
                time=0.0,
                dt=1.0,
            )

    @pytest.mark.parametrize(
        "bad_value, error_type",
        [([1.0 / 9.0] * 9, TypeError), (torch.ones(9, dtype=torch.int64), TypeError)],
    )
    def test_fraction_must_be_floating_tensor(self, functor, bad_value, error_type):
        with pytest.raises(error_type):
            functor.forward({"land_use_fraction": bad_value}, time=0.0, dt=1.0)

    @pytest.mark.parametrize(
        "bad_value, match",
        [
            (torch.tensor([float("nan")] + [0.125] * 8), "finite"),
            (torch.tensor([-0.1] + [0.1375] * 8), "non-negative"),
        ],
    )
    def test_fraction_finite_and_nonnegative(self, functor, bad_value, match):
        with pytest.raises(ValueError, match=match):
            functor.forward({"land_use_fraction": bad_value}, time=0.0, dt=1.0)

    def test_fraction_class_count_is_exact(self, functor):
        with pytest.raises(ValueError, match="exactly 9"):
            functor.forward(
                {"land_use_fraction": torch.full((8,), 1.0 / 8.0)},
                time=0.0,
                dt=1.0,
            )

    def test_fraction_closure_is_required(self, functor):
        with pytest.raises(ValueError, match="sum to one"):
            functor.forward(
                {"land_use_fraction": torch.full((9,), 0.2)},
                time=0.0,
                dt=1.0,
            )

    @pytest.mark.parametrize("driver", sorted(LandUseTransitions.DRIVER_INPUTS))
    def test_optional_drivers_are_finite_nonnegative_tensors(self, functor, driver):
        base = {"land_use_fraction": torch.full((9,), 1.0 / 9.0)}
        with pytest.raises(TypeError, match=driver):
            functor.forward({**base, driver: 1.0}, time=0.0, dt=1.0)
        with pytest.raises(ValueError, match=driver):
            functor.forward(
                {**base, driver: torch.tensor(float("nan"))},
                time=0.0,
                dt=1.0,
            )
        with pytest.raises(ValueError, match=driver):
            functor.forward(
                {**base, driver: torch.tensor(-0.1)},
                time=0.0,
                dt=1.0,
            )

    @pytest.mark.parametrize("driver", ["market_accessibility", "policy_intensity"])
    def test_bounded_drivers_reject_values_above_one(self, functor, driver):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            functor.forward(
                {
                    "land_use_fraction": torch.full((9,), 1.0 / 9.0),
                    driver: torch.tensor(1.1),
                },
                time=0.0,
                dt=1.0,
            )

    @pytest.mark.parametrize("bad_time_index", [torch.tensor(-1.0), torch.tensor(1.5)])
    def test_time_index_must_be_nonnegative_integer_scalar(self, functor, bad_time_index):
        with pytest.raises(ValueError, match="time_index"):
            functor.forward(
                {
                    "land_use_fraction": torch.full((9,), 1.0 / 9.0),
                    "time_index": bad_time_index,
                },
                time=0.0,
                dt=1.0,
            )

    def test_parameter_bounds_validation(self, functor):
        functor._validate_parameter_bounds("fraction", 0.0)
        functor._validate_parameter_bounds("fraction", 1.0)
        with pytest.raises(ValueError, match=r"\[0,1\]"):
            functor._validate_parameter_bounds("fraction", -0.1)
        with pytest.raises(ValueError, match=r"\[0,1\]"):
            functor._validate_parameter_bounds("fraction", 1.1)

    def test_generator_has_ctmc_structure(self, functor, sample_state):
        generator = functor._compute_transition_matrix(sample_state)
        assert generator.shape == (9, 9)
        assert torch.isfinite(generator).all()
        diagonal = torch.diagonal(generator)
        off_diagonal = generator - torch.diag(diagonal)
        assert torch.all(diagonal <= 0.0)
        assert torch.all(off_diagonal >= 0.0)
        assert torch.allclose(generator.sum(dim=1), torch.zeros(9, dtype=generator.dtype))

    def test_generator_rejects_invalid_rates(self, functor):
        negative = torch.zeros(9, 9)
        negative[0, 1] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            functor._build_ctmc_generator(negative)
        nonfinite = torch.zeros(9, 9)
        nonfinite[0, 1] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            functor._build_ctmc_generator(nonfinite)

    def test_channel_fractions_fail_instead_of_normalizing(self, functor):
        valid = functor._validate_channel_fractions([0.25, 0.75], "channel")
        stacked = torch.stack(valid)
        assert torch.allclose(
            stacked,
            torch.tensor([0.25, 0.75], dtype=stacked.dtype),
        )
        with pytest.raises(ValueError, match="sum to one"):
            functor._validate_channel_fractions([0.2, 0.2], "channel")
        with pytest.raises(ValueError, match="negative"):
            functor._validate_channel_fractions([-0.1, 1.1], "channel")

    def test_driver_scaling_changes_generator_without_breaking_closure(self, functor):
        fractions = torch.full((9,), 1.0 / 9.0)
        base = functor._compute_transition_matrix({"land_use_fraction": fractions})
        driven = functor._compute_transition_matrix(
            {
                "land_use_fraction": fractions,
                "gdp_per_capita": torch.tensor(50000.0),
                "population_density": torch.tensor(1000.0),
                "market_accessibility": torch.tensor(1.0),
            }
        )
        assert torch.sum(torch.abs(driven)) > torch.sum(torch.abs(base))
        assert torch.allclose(driven.sum(dim=1), torch.zeros(9))

    def test_matrix_exponential_conserves_fraction(self, functor):
        fractions = _normalized((4, 9), type_axis=1)
        delta = functor._apply_transition_matrix(fractions, dt=86400.0)
        updated = fractions + delta
        assert torch.isfinite(updated).all()
        assert torch.all(updated >= 0.0)
        assert torch.allclose(updated.sum(dim=1), torch.ones(4, dtype=updated.dtype))

    def test_transition_event_validation(self, functor):
        valid = TransitionEvent(
            LandUseType.PRIMARY_FOREST,
            LandUseType.CROPLAND_C3,
            area_fraction=0.1,
            time_index=2,
        )
        functor.add_transition_event(valid)
        assert functor.transition_events == [valid]
        with pytest.raises(ValueError, match=r"\[0,1\]"):
            functor.add_transition_event(
                TransitionEvent(
                    LandUseType.PRIMARY_FOREST,
                    LandUseType.CROPLAND_C3,
                    area_fraction=1.1,
                )
            )

    def test_transition_event_moves_fraction_at_matching_time(self, functor):
        functor.transition_matrix = torch.zeros_like(functor.transition_matrix)
        functor.add_transition_event(
            TransitionEvent(
                LandUseType.PRIMARY_FOREST,
                LandUseType.CROPLAND_C3,
                area_fraction=0.25,
                time_index=4,
            )
        )
        fractions = torch.zeros(9)
        fractions[0] = 1.0
        output = functor.forward(
            {"land_use_fraction": fractions, "time_index": torch.tensor(4.0)},
            time=0.0,
            dt=1.0,
        )
        updated = output["land_use_fraction"].value
        assert updated[0] == pytest.approx(0.75)
        assert updated[2] == pytest.approx(0.25)
        assert updated.sum() == pytest.approx(1.0)

    def test_spatial_event_mask_preserves_each_support(self, functor):
        functor.transition_matrix = torch.zeros_like(functor.transition_matrix)
        mask = torch.tensor([[1.0, 0.0], [0.5, 0.25]])
        functor.add_transition_event(
            TransitionEvent(
                LandUseType.PRIMARY_FOREST,
                LandUseType.SECONDARY_FOREST,
                area_fraction=0.2,
                time_index=1,
                spatial_mask=mask,
            )
        )
        fractions = torch.zeros(9, 2, 2)
        fractions[0] = 1.0
        output = functor.forward(
            {"land_use_fraction": fractions, "time_index": torch.tensor(1.0)},
            time=0.0,
            dt=1.0,
        )
        updated = output["land_use_fraction"].value
        assert torch.allclose(updated[1], 0.2 * mask)
        assert torch.allclose(updated.sum(dim=0), torch.ones(2, 2))

    def test_forward_returns_exact_explicit_update_contract(self, functor, sample_state):
        output = functor.forward(sample_state, time=0.0, dt=3600.0)
        assert set(output) == set(CANONICAL_OUTPUT_UNITS)
        for name, item in output.items():
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == CANONICAL_OUTPUT_UNITS[name]
            assert item.source_functor == functor.name
            assert torch.isfinite(item.value).all()

    def test_zero_dt_is_exact_identity(self, functor, sample_state):
        output = functor.forward(sample_state, time=0.0, dt=0.0)
        assert torch.equal(output["land_use_fraction"].value, sample_state["land_use_fraction"])
        assert torch.count_nonzero(
            output["diagnostics.land_use_transitions.delta_fraction"].value
        ) == 0

    @pytest.mark.parametrize("bad_dt", [-1.0, float("nan"), float("inf")])
    def test_dt_must_be_finite_and_nonnegative(self, functor, sample_state, bad_dt):
        with pytest.raises(ValueError, match="dt"):
            functor.forward(sample_state, time=0.0, dt=bad_dt)

    def test_conservation_ledger_uses_only_fraction_domains(self, functor, sample_state):
        output = functor.forward(sample_state, time=0.0, dt=3600.0)
        ledger = functor.check_conservation_laws(sample_state, output)
        assert set(ledger) == {
            "area_fraction_conservation",
            "area_increment_conservation",
        }
        assert max(ledger.values()) <= 1e-12

    def test_conservation_violation_fails_fast(self, functor, sample_state):
        with patch.object(
            functor,
            "_check_conservation_laws",
            return_value={"area_fraction_conservation": 1.0},
        ):
            with pytest.raises(ValueError, match="conservation violation"):
                functor.forward(sample_state, time=0.0, dt=3600.0)

    def test_unrelated_area_magnitude_does_not_change_result(self, functor, sample_state):
        small = dict(sample_state)
        large = dict(sample_state)
        small["external_area_observation"] = torch.ones(2, 9, 4, 5)
        large["external_area_observation"] = torch.full((2, 9, 4, 5), 1.0e12)
        small_output = functor.forward(small, time=0.0, dt=3600.0)
        large_output = functor.forward(large, time=0.0, dt=3600.0)
        for key in CANONICAL_OUTPUT_UNITS:
            assert torch.equal(small_output[key].value, large_output[key].value)

    def test_forward_does_not_mutate_input_or_keep_hidden_state(self, functor, sample_state):
        before = {key: value.clone() for key, value in sample_state.items()}
        functor.forward(sample_state, time=0.0, dt=3600.0)
        for key, value in before.items():
            assert torch.equal(sample_state[key], value)
        assert not hasattr(functor, "current_fractions")

    def test_dependency_declaration_is_empty(self, functor):
        assert functor.get_dependencies() == []
        assert functor._dependencies == []

    def test_bridge_and_torch_module_compliance(self, functor):
        assert isinstance(functor, ScaleCouplingFunctor)
        assert isinstance(functor, ScaleCoupling)
        assert isinstance(functor, torch.nn.Module)

    def test_compute_delegates_to_canonical_forward(self, functor, sample_state):
        direct = functor.forward(sample_state, time=0.0, dt=3600.0)
        computed = functor.compute(sample_state, time=0.0, dt=3600.0)
        assert set(computed) == set(direct)
        for key in direct:
            assert torch.equal(computed[key].value, direct[key].value)

    def test_seconds_per_year_uses_physical_constant_ssot(self, functor):
        fractions = torch.full((9,), 1.0 / 9.0)
        generator = torch.zeros(9, 9)
        with patch(
            "relic.scientific.anthroposphere.land_use.transitions.get_physical_constant",
            return_value=100.0,
        ) as accessor:
            delta = functor._apply_transition_matrix(
                fractions,
                dt=25.0,
                transition_matrix=generator,
            )
        accessor.assert_called_once_with("seconds_per_year")
        assert torch.equal(delta, torch.zeros_like(fractions))

    def test_removed_secondary_algorithms_are_not_exposed(self, functor):
        removed = {
            "_calculate_transition_probability",
            "_calculate_area_conversion",
            "_apply_spatial_influence",
            "_apply_temporal_dynamics",
            "_calculate_transition_rate",
            "_calculate_spatial_autocorrelation",
            "_compute_biomass_and_soil_changes",
            "_process_aggregate_mode",
            "_process_spatial_mode",
            "get_land_use_statistics",
        }
        assert not any(hasattr(functor, name) for name in removed)

    def test_forward_takes_the_state_only(self, functor):
        import inspect

        parameters = inspect.signature(type(functor).forward).parameters
        assert list(parameters) == ["self", "state", "time", "dt"]

    def test_spatial_output_shapes_follow_canonical_support(self, functor, sample_state):
        output = functor.forward(sample_state, time=0.0, dt=3600.0)
        assert output["land_use_fraction"].value.shape == (2, 9, 4, 5)
        assert output["diagnostics.land_use_transitions.delta_fraction"].value.shape == (
            2,
            9,
            4,
            5,
        )
        for key in (
            "diagnostics.land_use_transitions.dominant_type_index",
            "diagnostics.land_use_transitions.maximum_fraction",
            "diagnostics.land_use_transitions.soil_carbon_input_rate_change",
        ):
            assert output[key].value.shape == (2, 4, 5)

    def test_real_simulator_commits_full_state_with_canonical_provenance(self):
        fractions = torch.tensor(
            [0.25, 0.15, 0.20, 0.05, 0.15, 0.10, 0.05, 0.03, 0.02],
            dtype=torch.float64,
        )
        state = {
            "land_use_fraction": fractions.clone(),
            "time_index": torch.tensor(0, dtype=torch.int64),
            "gdp_per_capita": torch.tensor(250.0, dtype=torch.float64),
            "population_density": torch.tensor(150.0, dtype=torch.float64),
            "market_accessibility": torch.tensor(0.4, dtype=torch.float64),
            "policy_intensity": torch.tensor(0.2, dtype=torch.float64),
        }
        dt = 86400.0
        expected = LandUseTransitions().forward(state, time=0.0, dt=dt)
        simulator = RELICSimulator(
            {
                "max_time": dt,
                "time_step": dt,
                "conservation_checks": False,
                "strict_dependencies": False,
                "device": "cpu",
            }
        )
        simulator.initialize_state(state)
        runtime_functor = LandUseTransitions()
        simulator.register_functor(runtime_functor, execution_group="fast")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            outputs = simulator.step(dt=dt)

        assert set(outputs) == set(CANONICAL_OUTPUT_UNITS)
        final_state = simulator.state_manager.get_state()
        actual = final_state["land_use_fraction"]
        torch.testing.assert_close(
            actual,
            expected["land_use_fraction"].value,
            rtol=0.0,
            atol=0.0,
        )
        assert torch.max(torch.abs(actual - fractions)) > 0.0
        assert torch.abs(actual.sum() - 1.0) <= 1.0e-12
        spec = simulator.state_manager.get_variable_spec("land_use_fraction")
        assert spec is not None
        assert spec.units == "1"
        provenance = simulator.get_variable_provenance("land_use_fraction")
        assert {entry["functor"] for entry in provenance} == {runtime_functor.name}
        assert {entry["role"] for entry in provenance} == {"writer", "final_writer"}
        assert {entry["mode"] for entry in provenance} == {"UpdateMode.REPLACE"}
        for name in CANONICAL_OUTPUT_UNITS:
            if name.startswith("diagnostics."):
                assert name not in final_state
                assert simulator.get_variable_provenance(name) == []
        assert simulator.get_effective_device(runtime_functor.name).type == "cpu"
        assert caught == []
