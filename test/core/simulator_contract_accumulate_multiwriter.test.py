import pytest
import torch

from relic.core.functor import ExecutionPhase, FunctorMetadata, ProcessCategory
from relic.core.functor import TransportMechanismFunctor
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateMode


class _NitrogenLedgerFunctor(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(
        self,
        name: str,
        value: float,
        mode: UpdateMode,
        include_diagnostic: bool = False,
        output_name: str = "nh4_nitrogen",
    ):
        output_variables = [output_name]
        if include_diagnostic:
            output_variables.append("diagnostics.test.transaction_marker")
        metadata = FunctorMetadata(
            name=name,
            version="0.0.0",
            description="Contract-aware ammonium ledger update test functor.",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="nh4_new = nh4_old + delta_nh4",
            references=["test"],
            input_variables=[output_name],
            output_variables=output_variables,
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self._value = torch.tensor(value, dtype=torch.float32)
        self._mode = mode
        self._include_diagnostic = include_diagnostic
        self._output_name = output_name

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        updates = {
            self._output_name: self.wrap_update(self._value, mode=self._mode)
        }
        if self._include_diagnostic:
            updates["diagnostics.test.transaction_marker"] = torch.tensor(1.0)
        return updates


def test_contract_accumulate_accepts_authorized_multiwriters():
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"nh4_nitrogen": torch.tensor(0.3, dtype=torch.float32)})

    # Authorized writers from the canonical ammonium state ledger contract.
    sim.register_functor(
        _NitrogenLedgerFunctor("mineralization", 0.05, UpdateMode.ACCUMULATE),
        execution_group="fast",
    )
    sim.register_functor(
        _NitrogenLedgerFunctor("nitrification", 0.02, UpdateMode.ACCUMULATE),
        execution_group="fast",
    )

    sim.step(dt=1.0)
    nh4_nitrogen = sim.state_manager.get_state("nh4_nitrogen")
    assert torch.isclose(nh4_nitrogen, torch.tensor(0.37, dtype=torch.float32))


def test_contract_replace_rejects_unauthorized_writer():
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"nh4_nitrogen": torch.tensor(0.3, dtype=torch.float32)})
    sim.register_functor(
        _NitrogenLedgerFunctor(
            "unauthorized_replace_writer",
            0.05,
            UpdateMode.REPLACE,
            include_diagnostic=True,
        ),
        execution_group="fast",
    )

    state_before = sim.state_manager.get_state("nh4_nitrogen")
    counter_before = sim.state_manager.get_diagnostics()["update_counter"]
    history_before = len(sim.state_manager.conservation_history)
    diagnostics_before = dict(sim.state_manager._current_diagnostics)

    with pytest.raises(PermissionError):
        sim.step(dt=1.0)

    assert torch.equal(
        sim.state_manager.get_state("nh4_nitrogen"),
        state_before,
    )
    assert sim.state_manager.get_diagnostics()["update_counter"] == counter_before
    assert len(sim.state_manager.conservation_history) == history_before
    assert sim.state_manager._current_diagnostics == diagnostics_before
    assert sim.get_variable_provenance("nh4_nitrogen") == []
    assert "nh4_nitrogen" not in sim._step_update_sources
    assert "nh4_nitrogen" not in sim._step_functor_increments
    assert "nh4_nitrogen" not in getattr(sim, "_step_provenance_detail", {})
    assert sim._active_composition is None


def test_variable_provenance_distinguishes_final_writer_and_increment_sources():
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"nh4_nitrogen": torch.tensor(0.3, dtype=torch.float32)})
    sim.register_functor(
        _NitrogenLedgerFunctor("mineralization", 0.01, UpdateMode.ACCUMULATE),
        execution_group="fast",
    )
    sim.register_functor(
        _NitrogenLedgerFunctor("nitrification", 0.02, UpdateMode.ACCUMULATE),
        execution_group="fast",
    )

    sim.step(dt=1.0)
    records = sim.get_variable_provenance("nh4_nitrogen")
    roles = {item.get("role") for item in records}
    increment_sources = {
        item["functor"]
        for item in records
        if item.get("role") == "increment_source"
    }

    assert "final_writer" in roles
    assert "increment_source" in roles
    assert increment_sources == {"mineralization", "nitrification"}


def test_replace_writer_is_not_reported_as_accumulate_increment_source():
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state(
        {"net_primary_productivity": torch.tensor(0.0, dtype=torch.float32)}
    )
    sim.register_functor(
        _NitrogenLedgerFunctor(
            "autotrophic_respiration",
            0.25,
            UpdateMode.REPLACE,
            output_name="net_primary_productivity",
        ),
        execution_group="fast",
    )

    sim.step(dt=1.0)

    records = sim.get_variable_provenance("net_primary_productivity")
    assert {record["role"] for record in records} == {"writer", "final_writer"}
    assert all(record["mode"] == str(UpdateMode.REPLACE) for record in records)
    assert "net_primary_productivity" not in sim._step_functor_increments
