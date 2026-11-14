"""Tests for StateManager.apply_increment."""

import pytest
import torch
import relic.data.standards.unit_converter as unit_converter_module

from relic.core.state import StateManager, UpdateItem, UpdateMode, StateVariableSpec


def _build_state_manager() -> StateManager:
    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name="mass_pool",
            units="kg",
            conserved=True,
            default_value=1.0,
        )
    )
    return sm


def test_apply_increment_unit_conversion() -> None:
    sm = _build_state_manager()
    inc = UpdateItem(value=1000.0, unit="g", basis="per_step")
    updated = sm.apply_increment("mass_pool", inc)
    assert torch.is_tensor(updated)
    assert pytest.approx(updated.item(), rel=1e-6) == 2.0


def test_apply_increment_reuses_canonical_unit_converter(monkeypatch) -> None:
    sm = _build_state_manager()

    def reject_second_owner():
        raise AssertionError("UnitConverter runtime owner must be reused")

    monkeypatch.setattr(unit_converter_module, "UnitConverter", reject_second_owner)
    updated = sm.apply_increment(
        "mass_pool",
        UpdateItem(value=1000.0, unit="g", basis="per_step"),
    )

    assert updated.item() == pytest.approx(2.0)


def test_apply_increment_per_second_scaling() -> None:
    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name="mass_pool",
            units="kg",
            conserved=True,
            default_value=0.0,
        )
    )
    inc = UpdateItem(value=2.0, unit="kg", basis="per_second")
    updated = sm.apply_increment("mass_pool", inc, dt=10.0)
    assert pytest.approx(updated.item(), rel=1e-6) == 20.0


def test_apply_increment_requires_dt_for_per_second() -> None:
    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name="mass_pool",
            units="kg",
            conserved=True,
            default_value=0.0,
        )
    )
    inc = UpdateItem(value=1.0, unit="kg", basis="per_second")
    with pytest.raises(ValueError):
        sm.apply_increment("mass_pool", inc)


def test_apply_increment_rejects_inconsistent_units() -> None:
    sm = _build_state_manager()
    inc = UpdateItem(value=1.0, unit="K", basis="per_step")
    with pytest.raises(ValueError):
        sm.apply_increment("mass_pool", inc)

def test_dimensionless_state_accepts_a_per_second_tendency_labelled_s_minus_one() -> None:
    """A tendency of a unit-one state carries s-1 and integrates by dt (residue cover case)."""
    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name="residue_cover",
            units="dimensionless",
            conserved=False,
            default_value=0.5,
        )
    )
    # References: BIPM SI Brochure (9th ed.) Section 2.3.3; LeVeque (2002).
    # Mathematical expression: x_new = x_old + q dt with [x] = 1 and [q] = s-1.
    # Parameter meanings: q is the residue cover tendency, dt the step.
    # Units: x [1]; q [s-1]; dt [s].
    # Domain/constraints: ACCUMULATE rate item with per_timestep=False.
    # Sign convention: negative q lowers the cover.
    sm.apply_updates(
        {
            "delta_residue_cover": UpdateItem(
                value=torch.tensor(-1.0e-5),
                unit="s-1",
                mode=UpdateMode.ACCUMULATE,
                per_timestep=False,
            )
        },
        dt=1000.0,
        check_conservation=False,
    )
    assert sm.get_state("residue_cover").item() == pytest.approx(0.49, rel=1.0e-9)


def test_rate_increment_with_matching_rate_label_needs_no_identity_conversion() -> None:
    """A per-second ACCUMULATE increment whose label already is the rate label is integrated as is."""
    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name="mass_pool",
            units="kg m-2",
            conserved=True,
            default_value=1.0,
        )
    )
    sm.apply_updates(
        {
            "mass_pool": UpdateItem(
                value=torch.tensor(0.002),
                unit="kg m-2 s-1",
                mode=UpdateMode.ACCUMULATE,
                basis="per_second",
                per_timestep=False,
            )
        },
        dt=10.0,
        check_conservation=False,
    )
    assert sm.get_state("mass_pool").item() == pytest.approx(1.02, rel=1.0e-9)


def test_true_rate_unit_mismatch_still_raises() -> None:
    """A tendency in a different dimension is still rejected."""
    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name="mass_pool",
            units="kg m-2",
            conserved=True,
            default_value=1.0,
        )
    )
    with pytest.raises(ValueError, match="Unit mismatch"):
        sm.apply_updates(
            {
                "mass_pool": UpdateItem(
                    value=torch.tensor(1.0),
                    unit="K s-1",
                    mode=UpdateMode.ACCUMULATE,
                    basis="per_second",
                    per_timestep=False,
                )
            },
            dt=10.0,
            check_conservation=False,
        )

def test_registered_alias_labels_are_harmonized_without_an_identity_error() -> None:
    """A writer labelling a unit-one state as "1" must commit onto a "dimensionless" schema."""
    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name="scale_factor",
            units="dimensionless",
            conserved=False,
            default_value=1.0,
        )
    )
    sm.apply_updates(
        {
            "scale_factor": UpdateItem(
                value=torch.tensor(0.7),
                unit="1",
                mode=UpdateMode.REPLACE,
                per_timestep=False,
            )
        },
        dt=1.0,
        check_conservation=False,
    )
    assert sm.get_state("scale_factor").item() == pytest.approx(0.7)

