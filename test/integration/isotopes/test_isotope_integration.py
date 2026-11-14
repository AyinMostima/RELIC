"""Fail-closed integration tests for the runtime-unbound isotope shell."""

import inspect

import pytest
import torch

from relic.contracts.contract_loader import load_contract_registry
from relic.core.simulator import RELICSimulator
from relic.scientific.isotopes.config import get_default_isotope_config
import relic.scientific.isotopes.registry as isotope_registry
from relic.scientific.isotopes.registry import (
    IsotopeValidationError,
    wire_isotope_tracker,
)


def _make_real_simulator(isotope_config):
    """Create a real simulator without claiming isotope runtime support."""
    initial_state = {
        pool.state_variable: torch.tensor(100.0)
        for species_config in isotope_config.species_configs.values()
        for pool in species_config.pools.values()
    }
    initial_state["temperature"] = torch.tensor(298.15)
    # Pools without an SCM contract take the units the isotope configuration
    # declares for them; contracted pools keep their contract declaration.
    registry = load_contract_registry()
    state_units = {
        pool.state_variable: pool.units
        for species_config in isotope_config.species_configs.values()
        for pool in species_config.pools.values()
        if registry.to_state_variable_spec(pool.state_variable) is None
    }
    simulator = RELICSimulator(
        {"conservation_checks": False, "time_step": 3600.0, "state_units": state_units}
    )
    simulator.initialize_state(initial_state)
    return simulator


def _make_unsafe_runtime_config():
    """Use Pydantic's explicitly unsafe constructor for a boundary oracle."""
    config = get_default_isotope_config()
    species_configs = {
        species: type(species_config).model_construct(
            **{
                **species_config.__dict__,
                "enabled": True,
                "runtime_blocker": "",
            }
        )
        for species, species_config in config.species_configs.items()
    }
    return type(config).model_construct(
        **{
            **config.__dict__,
            "enabled": True,
            "tracked_species": frozenset(species_configs),
            "species_configs": species_configs,
        }
    )


def _assert_heartbeat_unregistered(simulator):
    """Assert that every simulator-owned registration surface is clean."""
    functor_name = "isotope_heartbeat_functor"
    assert functor_name not in simulator.functors
    assert all(
        functor_name not in members for members in simulator._groups.values()
    )
    assert functor_name not in (simulator.execution_order or [])
    assert functor_name not in simulator.dependency_graph.nodes


class TestIsotopeIntegration:
    """Public isotope assembly must remain side-effect free and [N]."""

    def test_public_default_config_fails_closed_before_registration(self):
        config = get_default_isotope_config()
        simulator = _make_real_simulator(config)

        with pytest.raises(IsotopeValidationError, match="root runtime_unbound"):
            wire_isotope_tracker(
                simulator,
                simulator.state_manager,
                config,
                enable_netcdf_output=False,
            )

        _assert_heartbeat_unregistered(simulator)

    def test_public_default_rejects_before_state_manager_read(
        self,
        monkeypatch,
    ):
        config = get_default_isotope_config()
        simulator = _make_real_simulator(config)
        state_reads = []

        def _forbid_state_read():
            state_reads.append(True)
            raise AssertionError("StateManager must not be read")

        monkeypatch.setattr(
            simulator.state_manager,
            "get_state",
            _forbid_state_read,
        )

        with pytest.raises(IsotopeValidationError, match="runtime_unbound"):
            wire_isotope_tracker(
                simulator,
                simulator.state_manager,
                config,
                enable_netcdf_output=False,
            )

        assert state_reads == []
        _assert_heartbeat_unregistered(simulator)

    def test_registry_revalidates_unsafe_config_before_state_manager_read(
        self,
        monkeypatch,
    ):
        unsafe_config = _make_unsafe_runtime_config()
        simulator = _make_real_simulator(unsafe_config)
        state_reads = []

        def _forbid_state_read():
            state_reads.append(True)
            raise AssertionError("StateManager must not be read")

        monkeypatch.setattr(
            simulator.state_manager,
            "get_state",
            _forbid_state_read,
        )

        with pytest.raises(IsotopeValidationError, match="root runtime_unbound"):
            wire_isotope_tracker(
                simulator,
                simulator.state_manager,
                unsafe_config,
                enable_netcdf_output=False,
            )

        assert state_reads == []
        _assert_heartbeat_unregistered(simulator)

    def test_registry_has_no_validation_bypass_or_parallel_error_taxonomy(self):
        parameters = inspect.signature(wire_isotope_tracker).parameters

        assert "validate_integration" not in parameters
        assert parameters["enable_netcdf_output"].default is False
        assert not hasattr(isotope_registry, "ValidationError")
        assert not hasattr(isotope_registry, "ServiceCoordinationError")
        assert not hasattr(isotope_registry, "IsotopeIntegrationError")
        assert not hasattr(isotope_registry, "IsotopeServiceCoordinationError")
        assert issubclass(isotope_registry.IsotopeValidationError, ValueError)

    def test_registry_exposes_only_one_fail_closed_assembly_entry(self):
        forbidden_public_helpers = (
            "IsotopeSystemStatus",
            "remove_isotope_tracker",
            "create_isotope_components",
            "get_integration_status",
            "quick_isotope_integration",
            "register_components",
            "validate_integration_requirements",
            "validate_isotope_integration",
            "_create_isotope_components",
            "_rollback_domain_components",
            "_validate_integration_requirements",
        )

        assert callable(isotope_registry.wire_isotope_tracker)
        for helper_name in forbidden_public_helpers:
            assert not hasattr(isotope_registry, helper_name)

    def test_netcdf_output_is_runtime_unbound_and_side_effect_free(
        self,
        tmp_path,
    ):
        config = get_default_isotope_config()
        output_parent = tmp_path / "uncreated" / "isotopes"
        config.unified_output.output_file = str(
            output_parent / "stable_isotopes.nc"
        )
        simulator = _make_real_simulator(config)

        with pytest.raises(
            IsotopeValidationError,
            match="postcommit_writer_unbound",
        ):
            wire_isotope_tracker(
                simulator,
                simulator.state_manager,
                config,
                enable_netcdf_output=True,
            )

        assert output_parent.exists() is False
        _assert_heartbeat_unregistered(simulator)
