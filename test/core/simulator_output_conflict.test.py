import logging

import torch
import pytest

from relic.core.functor import ExecutionPhase, FunctorMetadata, ProcessCategory
from relic.core.functor import TransportMechanismFunctor
from relic.core.simulator import RELICSimulator
from relic.core.simulator import output_overlap_governance
from relic.core.state import UpdateMode


class _FixedUpdateFunctor(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name: str, variable: str, value: float, mode: UpdateMode):
        metadata = FunctorMetadata(
            name=name,
            version="0.0.0",
            description=f"Test functor that updates {variable} with a fixed payload.",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=f"{variable}_new = {variable}_old + delta",
            references=["test"],
            input_variables=[variable],
            output_variables=[variable],
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self._value = torch.tensor(value, dtype=torch.float32)
        self._mode = mode
        self._variable = variable

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {self._variable: self.wrap_update(self._value, mode=self._mode)}


def test_simulator_conflicting_replace_updates_raise():
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"soil_moisture": torch.tensor(0.3, dtype=torch.float32)})

    sim.register_functor(
        _FixedUpdateFunctor("replace_a", "soil_moisture", 0.1, UpdateMode.REPLACE),
        execution_group="fast",
    )
    with pytest.raises(ValueError, match="Mutually exclusive overlap variable"):
        sim.register_functor(
            _FixedUpdateFunctor("replace_b", "soil_moisture", 0.2, UpdateMode.REPLACE),
            execution_group="fast",
        )


def test_simulator_allows_multiple_accumulate_updates():
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"nh4_nitrogen": torch.tensor(0.3, dtype=torch.float32)})

    # The ammonium contract declares these names as additive state writers.
    sim.register_functor(
        _FixedUpdateFunctor("mineralization", "nh4_nitrogen", 0.05, UpdateMode.ACCUMULATE),
        execution_group="fast",
    )
    sim.register_functor(
        _FixedUpdateFunctor("nitrification", "nh4_nitrogen", 0.02, UpdateMode.ACCUMULATE),
        execution_group="fast",
    )

    sim.step(dt=1.0)
    nh4_nitrogen = sim.state_manager.get_state("nh4_nitrogen")
    assert torch.isclose(nh4_nitrogen, torch.tensor(0.37, dtype=torch.float32))


def test_governed_aggregated_overlap_is_not_reported_as_warning(caplog):
    """An accepted SCM-governed additive ledger is normal runtime behavior."""
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"nh4_nitrogen": torch.tensor(0.3, dtype=torch.float32)})

    with caplog.at_level(logging.WARNING):
        sim.register_functor(
            _FixedUpdateFunctor(
                "mineralization", "nh4_nitrogen", 0.05, UpdateMode.ACCUMULATE
            ),
            execution_group="fast",
        )
        sim.register_functor(
            _FixedUpdateFunctor(
                "nitrification", "nh4_nitrogen", -0.02, UpdateMode.ACCUMULATE
            ),
            execution_group="fast",
        )

    assert "Governed aggregated overlap accepted" not in caplog.text


def test_zero_state_increment_has_no_flux_direction_warning(caplog):
    """A zero increment is neither an input nor an output boundary flow."""
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"no3_nitrogen": torch.tensor(0.3, dtype=torch.float32)})
    sim.register_functor(
        _FixedUpdateFunctor(
            "nitrogen_leaching", "no3_nitrogen", 0.0, UpdateMode.ACCUMULATE
        ),
        execution_group="fast",
    )

    with caplog.at_level(logging.WARNING):
        sim.step(dt=1.0)

    direction_warnings = [
        record
        for record in caplog.records
        if "no3_nitrogen" in record.getMessage()
        and ("whitelist" in record.getMessage() or "direction" in record.getMessage())
    ]
    assert direction_warnings == []


def test_positive_replace_flux_uses_declared_output_direction(caplog):
    """A positive boundary-flux magnitude follows SCM writer direction, not sign."""
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state(
        {"nitrogen_leaching_flux": torch.tensor(0.0, dtype=torch.float32)}
    )
    sim.register_functor(
        _FixedUpdateFunctor(
            "nitrogen_leaching",
            "nitrogen_leaching_flux",
            1.0e-6,
            UpdateMode.REPLACE,
        ),
        execution_group="fast",
    )

    with caplog.at_level(logging.WARNING):
        sim.step(dt=1.0)

    direction_warnings = [
        record
        for record in caplog.records
        if "nitrogen_leaching_flux" in record.getMessage()
        and "direction" in record.getMessage()
    ]
    assert direction_warnings == []


def test_positive_aggregated_rate_uses_declared_output_direction(caplog):
    """An additive same-step flux rate still follows the SCM output route."""
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state({"n2o_flux": torch.tensor(0.0, dtype=torch.float32)})
    sim.register_functor(
        _FixedUpdateFunctor(
            "nitrification",
            "n2o_flux",
            1.0e-6,
            UpdateMode.ACCUMULATE,
        ),
        execution_group="fast",
    )

    with caplog.at_level(logging.WARNING):
        sim.step(dt=1.0)

    direction_warnings = [
        record
        for record in caplog.records
        if "n2o_flux" in record.getMessage()
        and "direction" in record.getMessage()
    ]
    assert direction_warnings == []


def test_flux_direction_contract_checks_each_tensor_element(caplog):
    """Net-positive sums must not hide a locally unauthorized negative increment."""
    sim = RELICSimulator({"max_time": 1.0, "time_step": 1.0, "conservation_checks": False})
    sim.initialize_state(
        {"no3_nitrogen": torch.tensor([0.3, 0.3], dtype=torch.float32)}
    )
    sim.register_functor(
        _FixedUpdateFunctor(
            "nitrification",
            "no3_nitrogen",
            [0.02, -0.01],
            UpdateMode.ACCUMULATE,
        ),
        execution_group="fast",
    )

    with caplog.at_level(logging.WARNING):
        sim.step(dt=1.0)

    direction_warnings = [
        record
        for record in caplog.records
        if "no3_nitrogen" in record.getMessage()
        and ("whitelist" in record.getMessage() or "direction" in record.getMessage())
    ]
    assert len(direction_warnings) == 1


def test_overlap_governance_exposes_only_executable_policies():
    policies = output_overlap_governance.load_output_overlap_policies()

    assert {entry["policy"] for entry in policies.values()} <= {
        "aggregated",
        "mutually_exclusive",
    }
    assert not hasattr(output_overlap_governance, "OVERLAP_POLICY_PRIMARY_WRITER")
    assert not hasattr(output_overlap_governance, "get_primary_writer_class")

@pytest.mark.parametrize(
    ("content", "match"),
    [
        (None, "not found|No such file"),
        ("{not json", "Expecting|JSON"),
        ("[]", "mapping"),
        ('{"x": {"policy": "primary_writer", "reason": "legacy"}}', "policy"),
        ('{"x": {"policy": "aggregated", "reason": ""}}', "reason"),
    ],
)
def test_overlap_policy_file_problems_raise(tmp_path, monkeypatch, content, match):
    """The canonical overlap policy file is read completely or the load fails."""
    policy_file = tmp_path / "output_overlap_whitelist.json"
    if content is not None:
        policy_file.write_text(content, encoding="utf-8")
    monkeypatch.setattr(output_overlap_governance, "_policy_file_path", lambda: policy_file)
    output_overlap_governance.load_output_overlap_policies.cache_clear()
    try:
        with pytest.raises((OSError, ValueError, TypeError), match=match):
            output_overlap_governance.load_output_overlap_policies()
    finally:
        output_overlap_governance.load_output_overlap_policies.cache_clear()
