import re
from pathlib import Path

import pytest

import relic.scientific.anthroposphere as anthroposphere_package
from relic.core.functor import ScaleCouplingFunctor
from relic.core.simulator import RELICSimulator
from relic.parameters import get_global_manager
from relic.scientific.anthroposphere import interface_mapping as mapping_module
from relic.scientific.anthroposphere.interface_mapping import (
    AnthroposphereInterfaceMapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_BLOCKER = (
    "[N] runtime_metadata_domain_unbound: anthroposphere CF metadata requires "
    "an exact SCM variable, official standard-name membership, physical-basis "
    "and coordinate binding, and a governed metadata application path"
)


class _ExplosiveState(dict):
    def __contains__(self, key):
        raise AssertionError(f"state membership was accessed: {key}")

    def __getitem__(self, key):
        raise AssertionError(f"state item was accessed: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"state get was accessed: {key}")

    def items(self):
        raise AssertionError("state items were accessed")


def test_interface_mapping_keeps_only_the_scale_coupling_identity():
    assert issubclass(AnthroposphereInterfaceMapping, ScaleCouplingFunctor)
    assert mapping_module.__all__ == ["AnthroposphereInterfaceMapping"]
    assert "AnthroposphereInterfaceMapping" in anthroposphere_package.__all__
    assert "AnthroposphereCFMapping" not in anthroposphere_package.__all__
    assert "AnthroposphereVariables" not in anthroposphere_package.__all__


def test_constructor_rejects_before_functor_or_metadata_effects(monkeypatch):
    base_calls = []

    def _reject_base_init(*args, **kwargs):
        base_calls.append((args, kwargs))
        raise AssertionError("FunctorBase initialization was reached")

    monkeypatch.setattr(ScaleCouplingFunctor, "__init__", _reject_base_init)

    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        AnthroposphereInterfaceMapping(
            name="forbidden",
            config={"strict_mode": False},
            strict_mode=False,
        )

    assert base_calls == []


@pytest.mark.parametrize(
    "args,kwargs",
    [
        pytest.param((), {}, id="default"),
        pytest.param(("mapping",), {}, id="positional-name"),
        pytest.param(
            (),
            {"name": "mapping", "config": {}, "strict_mode": True},
            id="keywords",
        ),
    ],
)
def test_constructor_has_one_deterministic_runtime_blocker(args, kwargs):
    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        AnthroposphereInterfaceMapping(*args, **kwargs)


def test_unsafe_constructor_bypass_cannot_scan_state_or_emit_diagnostics():
    mapping = object.__new__(AnthroposphereInterfaceMapping)
    state = _ExplosiveState()

    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        AnthroposphereInterfaceMapping.forward(
            mapping,
            state,
            time=0.0,
            dt=60.0,
            current_units={"co_emissions": "kg m-2 s-1"},
            dimensions={"co_emissions": ("y", "x")},
        )

    assert dict(state) == {}
    assert not hasattr(mapping, "_parameters")
    assert not hasattr(mapping, "cf_mapping")


def test_preloaded_tensor_cannot_unlock_the_unbound_metadata_runtime():
    mapping = object.__new__(AnthroposphereInterfaceMapping)
    state = {"co_emissions": object()}

    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        AnthroposphereInterfaceMapping.forward(
            mapping,
            state,
            time=0.0,
            dt=1.0,
        )


def test_simulator_registration_failure_is_atomic():
    simulator = RELICSimulator(
        {
            "time_step": 1.0,
            "max_time": 1.0,
            "calendar_type": "noleap",
            "device": "cpu",
            "conservation_checks": False,
            "strict_dependencies": True,
            "strict_scale_detection": False,
        }
    )
    before = simulator.get_functor_summary()

    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        simulator.register_functor(AnthroposphereInterfaceMapping())

    assert simulator.get_functor_summary() == before


def test_public_adapter_and_old_runtime_surfaces_are_absent():
    forbidden_module_api = {
        "AnthroposphereCFMapping",
        "AnthroposphereVariables",
    }
    forbidden_class_api = {
        "get_dependencies",
        "required_parameters",
        "compute",
        "map_variable",
        "map_multiple_variables",
        "sync_after_processing",
        "create_emission_metadata",
        "validate_mapping",
        "get_supported_variables",
        "declared_forms",
        "build_hodge_contract",
        "build_hodge_terms",
        "energy_terms",
        "entropy_terms",
        "reversible_structure",
        "dissipative_structure",
        "ports",
        "PHASE8_RUNTIME_REPLACEMENT_STATUS",
    }

    assert not [name for name in forbidden_module_api if hasattr(mapping_module, name)]
    assert forbidden_class_api.isdisjoint(AnthroposphereInterfaceMapping.__dict__)


def test_source_contains_only_the_fail_closed_boundary():
    source_path = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "anthroposphere"
        / "interface_mapping.py"
    )
    source = source_path.read_text(encoding="utf-8")
    forbidden_tokens = (
        "AnthroposphereCFMapping",
        "AnthroposphereVariables",
        "get_global_manager",
        "unit_converter",
        "map_variable",
        "standard_name.*",
        "strict_mode=False",
        "error_count",
        "success_count",
        "attempted_count",
        "state.items()",
        "diagnostics.anthroposphere.interface_mapping",
        "CochainSpaceSpec",
        "HodgeStarSpec",
        "ObservationOperatorSpec",
        "canonical_iphs_runtime",
    )

    assert "[N] runtime_metadata_domain_unbound" in source
    assert "official standard-name membership" in source
    assert "SCM" in source
    for token in forbidden_tokens:
        assert token not in source


def test_obsolete_positive_output_contract_is_removed():
    assert not (
        PROJECT_ROOT
        / "test"
        / "scientific"
        / "anthroposphere"
        / "interface_mapping"
        / "output_contract.spec.py"
    ).exists()


def test_orphan_generic_cf_parameter_catalog_is_removed():
    catalog_path = (
        PROJECT_ROOT
        / "relic"
        / "parameters"
        / "database"
        / "relic"
        / "anthroposphere"
        / "interface_parameters.yaml"
    )
    prefix = "anthroposphere.interface.mapping."
    representative_keys = (
        f"{prefix}standard_name.co_emissions",
        f"{prefix}standard_name.cropland_fraction",
        f"{prefix}standard_name.emission_field_3d",
        f"{prefix}standard_unit.mole_fraction_of_isoprene_in_air",
    )

    assert not catalog_path.exists()

    manager = get_global_manager(preload_all=True)
    active_keys = sorted(
        key for key in manager.get_all_parameters() if key.startswith(prefix)
    )
    assert active_keys == []
    for key in representative_keys:
        assert manager.get_parameter(key) is None
        with pytest.raises(KeyError):
            manager.get(key, required=True)
