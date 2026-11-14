import inspect

import pytest
import torch

from relic.coupling.core.conservation_checker import (
    check_energy_conservation,
    check_mass_conservation,
)
from relic.coupling.scales.conservation_enforcer import ConservationEnforcer


def test_mass_conservation_without_areas():
    source = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([6.0])
    result = check_mass_conservation(source, target, tolerance=1e-9, variable_name="flux")
    assert result["conserved"] == 1.0
    assert result["relative_error"] < 1e-9
    assert result["variable_name"] == "flux"


def test_mass_conservation_with_areas():
    source = torch.tensor([2.0, 2.0])
    target = torch.tensor([1.0, 1.0, 1.0, 1.0])
    source_areas = torch.tensor([1.0, 1.0])
    target_areas = torch.tensor([0.5, 0.5, 0.5, 0.5])
    result = check_mass_conservation(
        source,
        target,
        source_areas=source_areas,
        target_areas=target_areas,
        tolerance=1e-9,
    )
    assert result["conserved"] == 0.0
    assert result["relative_error"] > 0.0


def test_energy_conservation():
    energy_in = torch.tensor([10.0, 5.0])
    energy_out = torch.tensor([7.5, 7.5])
    result = check_energy_conservation(energy_in, energy_out, tolerance=1e-9)
    assert result["conserved"] == 1.0
    assert result["relative_error"] < 1e-9


def test_conservation_enforcer_delegates_checker():
    source = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.5, 1.5])
    enforcer = ConservationEnforcer(track_history=False)

    diag = enforcer.check_conservation(source, target, property_name="water", tolerance=1e-6, batch_ndim=0)
    reference = check_mass_conservation(
        source, target, tolerance=1e-6, variable_name="water"
    )

    assert abs(diag.relative_error - reference["relative_error"]) < 1e-9
    assert diag.tolerance == 1e-6
    assert diag.is_conserved == (reference["relative_error"] <= 1e-6)


def test_conservation_enforcer_resolves_no_tolerance_of_its_own():
    """The caller owns the ledger tolerance; the enforcer keeps, looks up and defaults none."""
    import relic.coupling.scales.conservation_enforcer as enforcer_module

    init_parameters = inspect.signature(ConservationEnforcer.__init__).parameters
    assert "tolerance" not in init_parameters
    assert "parameter_manager" not in init_parameters
    tolerance = inspect.signature(ConservationEnforcer.check_conservation).parameters["tolerance"]
    assert tolerance.kind is inspect.Parameter.KEYWORD_ONLY
    assert tolerance.default is inspect.Parameter.empty
    for name in ("get_conservation_tolerance", "get_precision_policy", "get_global_manager"):
        assert not hasattr(enforcer_module, name), name

    enforcer = ConservationEnforcer(track_history=False)
    assert not hasattr(enforcer, "tolerance")
    source = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.5, 1.5])
    with pytest.raises(TypeError):
        enforcer.check_conservation(source, target, property_name="water", batch_ndim=0)
    for invalid in (float("nan"), float("inf"), 0.0, 1.0, -1.0e-6, True):
        with pytest.raises(ValueError, match="tolerance"):
            enforcer.check_conservation(source, target, property_name="water", tolerance=invalid, batch_ndim=0)

