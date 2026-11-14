import warnings

import torch

from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.scientific.pedosphere.methane import (
    MethaneBudgetCoupler,
    MethaneOxidationFunctor,
    MethaneProductionFunctor,
    MethaneScaleCouplingFunctor,
    MethaneTransportFunctor,
)
from relic.scientific.pedosphere.methane.registry import register_methane_functors
from relic.utils.physical_constants import get_physical_constant


class _StandaloneMethaneProductionFunctor(MethaneProductionFunctor):
    """Use an externally committed gating control for provenance isolation."""

    def get_dependencies(self):
        return []


class _StandaloneMethaneOxidationFunctor(MethaneOxidationFunctor):
    """Use an externally committed gating control for provenance isolation."""

    def get_dependencies(self):
        return []


class _StandaloneMethaneBudgetCoupler(MethaneBudgetCoupler):
    """Use externally committed process ports for provenance isolation."""

    def get_dependencies(self):
        return []


def test_strict_registration_uses_single_storage_writers() -> None:
    simulator = RELICSimulator(config={"strict_dependencies": True, "device": "cpu"})
    register_methane_functors(simulator)

    storage_writers = sorted(
        name
        for name, functor in simulator.functors.items()
        if "ch4_soil_storage" in (getattr(functor.metadata, "output_variables", []) or [])
    )
    assert storage_writers == ["pedosphere_methane_budget"]
    assert all(
        "ch4_soil_storage_delta"
        not in (getattr(functor.metadata, "output_variables", []) or [])
        for functor in simulator.functors.values()
    )


def test_scale_ports_commit_with_conservation_and_real_provenance() -> None:
    state = {
        "land_surface.wetland_fraction": torch.tensor(0.65, dtype=torch.float64),
        "water_filled_pore_space": torch.tensor(0.70, dtype=torch.float64),
        "methane_data_score": torch.tensor(0.60, dtype=torch.float64),
    }
    dt = 1800.0
    scale = MethaneScaleCouplingFunctor()
    direct = scale.forward(
        {key: value.clone() for key, value in state.items()},
        time=0.0,
        dt=torch.tensor(dt, dtype=torch.float64),
    )
    state_manager = StateManager(
        device="cpu",
        conservation_checks={"enabled": True, "raise_on_violation": True},
    )
    state_manager.set_state(
        {key: value.clone() for key, value in state.items()},
        validate=False,
    )
    simulator = RELICSimulator(
        config={
            "strict_dependencies": True,
            "strict_scale_detection": False,
            "conservation_checks": True,
            "device": "cpu",
            "time_step": dt,
            "max_time": dt,
        }
    )
    simulator.initialize(state_manager, DependencyGraphResolver())
    simulator.register_functor(scale, execution_group="fast")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        updates = simulator.step(dt=dt)

    assert set(updates) == set(scale.metadata.output_variables)
    for variable in scale.metadata.output_variables:
        actual = state_manager.get_state(variable)
        torch.testing.assert_close(
            actual,
            direct[variable].value.to(dtype=actual.dtype),
            rtol=0.0,
            atol=1.0e-12,
        )
        writers = {
            item["functor"]
            for item in simulator.get_variable_provenance(variable)
            if item.get("role") in {"writer", "final_writer"}
        }
        assert writers == {"pedosphere_methane_scale"}
    assert str(simulator.get_effective_device(scale.name)) == "cpu"
    user_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
    ]
    assert not user_warnings, [str(warning.message) for warning in user_warnings]


def test_production_ports_have_real_simulator_provenance_after_one_step() -> None:
    seconds_per_day = float(get_physical_constant("seconds_per_day"))
    state = {
        "soil_temperature": torch.tensor(292.0),
        "soil_ph": torch.tensor(6.8),
        "heterotrophic_respiration": torch.tensor(5.5 / seconds_per_day),
        "redox_potential": torch.tensor(-190.0),
        "microbial_activity_index": torch.tensor(0.95),
        "land_surface.wetland_fraction": torch.tensor(0.65),
        "methane_gating_weight": torch.tensor(0.4),
    }

    dt = 1800.0
    production = _StandaloneMethaneProductionFunctor()
    direct = production.forward(
        {key: value.clone() for key, value in state.items()},
        time=0.0,
        dt=torch.tensor(dt),
    )
    state_manager = StateManager(
        device="cpu",
        conservation_checks={"enabled": False},
    )
    state_manager.set_state(
        {key: value.clone() for key, value in state.items()},
        validate=False,
    )
    simulator = RELICSimulator(
        config={
            "strict_dependencies": True,
            "strict_scale_detection": False,
            "conservation_checks": False,
            "device": "cpu",
            "time_step": dt,
            "max_time": dt,
        }
    )
    simulator.initialize(state_manager, DependencyGraphResolver())
    simulator.register_functor(production, execution_group="fast")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        updates = simulator.step(dt=dt)

    assert set(updates) == set(production.metadata.output_variables)
    for variable in ("ch4_production_rate", "ch4_storage_tendency_prod"):
        actual = state_manager.get_state(variable)
        torch.testing.assert_close(
            actual,
            direct[variable].value.to(dtype=actual.dtype),
            rtol=0.0,
            atol=1.0e-12,
        )
        provenance = simulator.get_variable_provenance(variable)
        writers = {
            item["functor"]
            for item in provenance
            if item.get("role") in {"writer", "final_writer"}
        }
        assert writers == {"pedosphere_methane_production"}
    assert str(simulator.get_effective_device("pedosphere_methane_production")) == "cpu"
    user_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
    ]
    assert not user_warnings, [str(warning.message) for warning in user_warnings]


def test_production_and_budget_commit_one_carbon_input_with_conservation() -> None:
    seconds_per_day = float(get_physical_constant("seconds_per_day"))
    state = {
        "soil_temperature": torch.tensor(292.0, dtype=torch.float64),
        "soil_ph": torch.tensor(6.8, dtype=torch.float64),
        "heterotrophic_respiration": torch.tensor(
            5.5 / seconds_per_day,
            dtype=torch.float64,
        ),
        "redox_potential": torch.tensor(-190.0, dtype=torch.float64),
        "microbial_activity_index": torch.tensor(0.95, dtype=torch.float64),
        "land_surface.wetland_fraction": torch.tensor(0.65, dtype=torch.float64),
        "methane_gating_weight": torch.tensor(0.4, dtype=torch.float64),
        "ch4_oxidation_rate": torch.tensor(0.0, dtype=torch.float64),
        "ch4_transport_loss_rate": torch.tensor(0.0, dtype=torch.float64),
        "ch4_storage_tendency_oxid": torch.tensor(0.0, dtype=torch.float64),
        "ch4_storage_tendency_trans": torch.tensor(0.0, dtype=torch.float64),
        "ch4_soil_storage": torch.tensor(0.0, dtype=torch.float64),
    }
    dt = 1800.0
    production = _StandaloneMethaneProductionFunctor()
    budget = _StandaloneMethaneBudgetCoupler()
    production_updates = production.forward(
        {key: value.clone() for key, value in state.items()},
        time=0.0,
        dt=torch.tensor(dt, dtype=torch.float64),
    )
    expected_production = production_updates["ch4_production_rate"].value
    budget_state = {key: value.clone() for key, value in state.items()}
    budget_state.update(
        {
            key: update.value
            for key, update in production_updates.items()
            if not key.startswith("diagnostics.")
        }
    )
    budget_updates = budget.forward(
        budget_state,
        time=0.0,
        dt=torch.tensor(dt, dtype=torch.float64),
    )
    direct_updates = {**production_updates, **budget_updates}
    state_manager = StateManager(
        device="cpu",
        conservation_checks={"enabled": True, "raise_on_violation": True},
    )
    state_manager.set_state(
        {key: value.clone() for key, value in state.items()},
        validate=False,
    )
    simulator = RELICSimulator(
        config={
            "strict_dependencies": True,
            "strict_scale_detection": False,
            "conservation_checks": True,
            "device": "cpu",
            "time_step": dt,
            "max_time": dt,
        }
    )
    simulator.initialize(state_manager, DependencyGraphResolver())
    simulator.register_functor(production, execution_group="fast")
    simulator.register_functor(budget, execution_group="fast")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulator.step(dt=dt)

    committed_production = state_manager.get_state("ch4_production_rate")
    actual_storage_delta = (
        state_manager.get_state("ch4_soil_storage") - state["ch4_soil_storage"]
    )
    expected_storage_delta = (
        committed_production
        * torch.as_tensor(dt, dtype=committed_production.dtype)
    ).to(dtype=actual_storage_delta.dtype)
    torch.testing.assert_close(
        actual_storage_delta,
        expected_storage_delta,
        rtol=0.0,
        atol=1.0e-12,
    )
    summary = state_manager.generate_conservation_report(
        direct_updates,
        current_time=dt,
        dt=dt,
        store_history=False,
    )
    assert summary.total_violations == 0
    assert summary.carbon_report is not None
    assert summary.carbon_report.balance_error <= 1.0e-12
    guardian = summary.custom_checks["conservation_guardian"]["carbon.total"]
    expected_production_scalar = float(
        expected_production.detach().reshape(-1)[0].tolist()
    )
    expected_storage_scalar = float(
        budget_updates["ch4_soil_storage"].value.detach().reshape(-1)[0].tolist()
    )
    assert abs(guardian["sum_flux_in"] - expected_production_scalar) <= 1.0e-12
    assert abs(
        guardian["delta_storage"]
        - expected_storage_scalar
    ) <= 1.0e-12
    assert guardian["sum_flux_out"] == 0.0
    assert guardian["residual"] == 0.0
    assert str(simulator.get_effective_device(production.name)) == "cpu"
    assert str(simulator.get_effective_device(budget.name)) == "cpu"
    for variable, expected_writer in {
        "ch4_production_rate": production.name,
        "ch4_storage_tendency_prod": production.name,
        "ch4_atmospheric_flux": budget.name,
        "ch4_soil_storage": budget.name,
    }.items():
        writers = {
            item["functor"]
            for item in simulator.get_variable_provenance(variable)
            if item.get("role") in {"writer", "final_writer"}
        }
        assert writers == {expected_writer}
    user_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
    ]
    assert not user_warnings, [str(warning.message) for warning in user_warnings]


def test_oxidation_ports_have_real_simulator_provenance_after_one_step() -> None:
    state = {
        "ch4_soil_storage": torch.tensor(10.0),
        "soil_temperature": torch.tensor(293.15),
        "soil_o2_concentration": torch.tensor(0.02),
        "porosity": torch.tensor(0.45),
        "water_filled_pore_space": torch.tensor(1.0),
    }
    dt = 1800.0
    oxidation = _StandaloneMethaneOxidationFunctor()
    direct = oxidation.forward(
        {key: value.clone() for key, value in state.items()},
        time=0.0,
        dt=torch.tensor(dt),
    )
    state_manager = StateManager(
        device="cpu",
        conservation_checks={"enabled": False},
    )
    state_manager.set_state(
        {key: value.clone() for key, value in state.items()},
        validate=False,
    )
    simulator = RELICSimulator(
        config={
            "strict_dependencies": True,
            "strict_scale_detection": False,
            "conservation_checks": False,
            "device": "cpu",
            "time_step": dt,
            "max_time": dt,
        }
    )
    simulator.initialize(state_manager, DependencyGraphResolver())
    simulator.register_functor(oxidation, execution_group="fast")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        updates = simulator.step(dt=dt)

    assert set(updates) == set(oxidation.metadata.output_variables)
    for variable in ("ch4_oxidation_rate", "ch4_storage_tendency_oxid"):
        actual = state_manager.get_state(variable)
        torch.testing.assert_close(
            actual,
            direct[variable].value.to(dtype=actual.dtype),
            rtol=0.0,
            atol=1.0e-12,
        )
        provenance = simulator.get_variable_provenance(variable)
        writers = {
            item["functor"]
            for item in provenance
            if item.get("role") in {"writer", "final_writer"}
        }
        assert writers == {"pedosphere_methane_oxidation"}
    assert str(simulator.get_effective_device("pedosphere_methane_oxidation")) == "cpu"
    user_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
    ]
    assert not user_warnings, [str(warning.message) for warning in user_warnings]


def test_transport_ports_commit_with_conservation_and_real_provenance() -> None:
    state = {
        "ch4_soil_storage": torch.tensor(3.0, dtype=torch.float64),
        "water_filled_pore_space": torch.tensor(0.4, dtype=torch.float64),
        "porosity": torch.tensor(0.45, dtype=torch.float64),
        "atmospheric_ch4_concentration": torch.tensor(0.0, dtype=torch.float64),
    }
    dt = 3600.0
    transport = MethaneTransportFunctor()
    direct = transport.forward(
        {key: value.clone() for key, value in state.items()},
        time=0.0,
        dt=torch.tensor(dt, dtype=torch.float64),
    )
    state_manager = StateManager(
        device="cpu",
        conservation_checks={"enabled": True, "raise_on_violation": True},
    )
    state_manager.set_state(
        {key: value.clone() for key, value in state.items()},
        validate=False,
    )
    simulator = RELICSimulator(
        config={
            "strict_dependencies": True,
            "strict_scale_detection": False,
            "conservation_checks": True,
            "device": "cpu",
            "time_step": dt,
            "max_time": dt,
        }
    )
    simulator.initialize(state_manager, DependencyGraphResolver())
    simulator.register_functor(transport, execution_group="fast")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        updates = simulator.step(dt=dt)

    assert set(updates) == set(transport.metadata.output_variables)
    for variable in ("ch4_transport_loss_rate", "ch4_storage_tendency_trans"):
        actual = state_manager.get_state(variable)
        torch.testing.assert_close(
            actual,
            direct[variable].value.to(dtype=actual.dtype),
            rtol=0.0,
            atol=1.0e-12,
        )
        provenance = simulator.get_variable_provenance(variable)
        writers = {
            item["functor"]
            for item in provenance
            if item.get("role") in {"writer", "final_writer"}
        }
        assert writers == {"pedosphere_methane_transport"}
    assert str(simulator.get_effective_device("pedosphere_methane_transport")) == "cpu"
    user_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
    ]
    assert not user_warnings, [str(warning.message) for warning in user_warnings]


def test_budget_ports_have_real_simulator_provenance_after_one_step() -> None:
    state = {
        "ch4_production_rate": torch.tensor(0.002, dtype=torch.float64),
        "ch4_oxidation_rate": torch.tensor(0.001, dtype=torch.float64),
        "ch4_transport_loss_rate": torch.tensor(0.0005, dtype=torch.float64),
        "ch4_storage_tendency_prod": torch.tensor(0.002, dtype=torch.float64),
        "ch4_storage_tendency_oxid": torch.tensor(-0.001, dtype=torch.float64),
        "ch4_storage_tendency_trans": torch.tensor(-0.0005, dtype=torch.float64),
        "ch4_soil_storage": torch.tensor(1.0, dtype=torch.float64),
    }
    dt = 1000.0
    budget = _StandaloneMethaneBudgetCoupler()
    direct = budget.forward(
        {key: value.clone() for key, value in state.items()},
        time=0.0,
        dt=torch.tensor(dt, dtype=torch.float64),
    )
    state_manager = StateManager(
        device="cpu",
        conservation_checks={"enabled": False},
    )
    state_manager.set_state(
        {key: value.clone() for key, value in state.items()},
        validate=False,
    )
    simulator = RELICSimulator(
        config={
            "strict_dependencies": True,
            "strict_scale_detection": False,
            "conservation_checks": False,
            "device": "cpu",
            "time_step": dt,
            "max_time": dt,
        }
    )
    simulator.initialize(state_manager, DependencyGraphResolver())
    simulator.register_functor(budget, execution_group="fast")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        updates = simulator.step(dt=dt)

    assert set(updates) == set(budget.metadata.output_variables)
    torch.testing.assert_close(
        state_manager.get_state("ch4_soil_storage") - state["ch4_soil_storage"],
        direct["ch4_soil_storage"].value,
        rtol=0.0,
        atol=1.0e-12,
    )
    for variable in ("ch4_atmospheric_flux", "ch4_soil_storage"):
        provenance = simulator.get_variable_provenance(variable)
        writers = {
            item["functor"]
            for item in provenance
            if item.get("role") in {"writer", "final_writer"}
        }
        assert writers == {"pedosphere_methane_budget"}
    assert str(simulator.get_effective_device("pedosphere_methane_budget")) == "cpu"
    user_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
    ]
    assert not user_warnings, [str(warning.message) for warning in user_warnings]


def test_methane_couplers_declare_their_coupling_scales() -> None:
    """Round 48a: each coupler states the two scales it couples; no bridge placeholder."""
    expected = {
        MethaneBudgetCoupler: ("pedosphere", "atmosphere", "spatial"),
        _StandaloneMethaneBudgetCoupler: ("pedosphere", "atmosphere", "spatial"),
        MethaneScaleCouplingFunctor: ("process", "grid_cell", "hierarchical"),
    }
    for coupler_class, (source, target, coupling) in expected.items():
        metadata = coupler_class().get_coupling_metadata()
        assert (
            metadata["source_scale"],
            metadata["target_scale"],
            metadata["coupling_type"],
        ) == (source, target, coupling), coupler_class.__name__
