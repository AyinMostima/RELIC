"""Owner tests of the whole-support reduction aggregate_spatial_alias (ledger D64, MA-21 item 122).

One law per declared basis (theory scale_coupling.md 2.2): intensive and area_density fields are
weighted means phi = sum_i w_i x_i / sum_i w_i with sum_i w_i > 0 (never max(sum_i w_i, 1)); cell
totals are summed; categorical codes take the weighted mode. The audit of every amount variable is
the ledger of the returned value: sum_i x_i w_i against phi * sum_i w_i for an area density and
sum_i X_i against the returned total for a cell total.
"""
from __future__ import annotations

import pytest
import torch

import relic.coupling.scales.alias_utils as alias_utils
from relic.coupling.core.policy import ScalingMode
from relic.coupling.scales.alias_utils import aggregate_spatial_alias
from relic.coupling.scales.types import ScalingStrategy, SpatialQuantityBasis

TOLERANCE = 1.0e-8


def _reduce(data, bases, *, weights=None, strategy=None, enforce=False, mode=ScalingMode.CONSERVATIVE):
    return aggregate_spatial_alias(
        data_dict=data,
        variable_bases={name: SpatialQuantityBasis.parse(basis) for name, basis in bases.items()},
        source_scale="grid_cell",
        target_scale="support",
        mode=mode,
        device=torch.device("cpu"),
        enforce_conservation=enforce,
        tolerance=TOLERANCE,
        weights=weights,
        strategy=strategy,
    )


def _f64(values):
    return torch.tensor(values, dtype=torch.float64)


def test_each_basis_follows_its_one_law_and_the_amounts_are_audited():
    data = {
        "rho": _f64([4.0, 8.0]),
        "total": _f64([4.0, 8.0]),
        "temperature": _f64([280.0, 290.0]),
        "code": _f64([3.0, 7.0]),
    }
    bases = {"rho": "area_density", "total": "cell_total", "temperature": "intensive", "code": "categorical"}
    result, audit = _reduce(data, bases, weights=_f64([1.0, 3.0]), enforce=True)
    # rho = (4 * 1 + 8 * 3) / 4 = 7 (a density stays a density); totals are summed: 4 + 8.
    assert float(result["rho"]) == pytest.approx(7.0)
    assert float(result["total"]) == pytest.approx(12.0)
    assert float(result["temperature"]) == pytest.approx(287.5)
    assert float(result["code"]) == pytest.approx(7.0)
    assert audit.metadata["output_basis"] == {name: basis for name, basis in bases.items()}
    assert audit.metadata["reduction"] == {
        "rho": "area_weighted_mean",
        "total": "sum",
        "temperature": "area_weighted_mean",
        "code": "area_weighted_mode",
    }
    assert audit.metadata["weights_applied"]["total"] is False
    assert audit.metadata["conservation_audited_variables"] == ["rho", "total"]
    rho = audit.metadata["conservation_audit"]["rho"]
    assert rho.source_total == pytest.approx(28.0) and rho.target_total == pytest.approx(28.0)
    assert rho.metadata["support_measure"] == pytest.approx(4.0)
    total = audit.metadata["conservation_audit"]["total"]
    assert total.source_total == pytest.approx(12.0) and total.target_total == pytest.approx(12.0)
    assert audit.relative_error < 1.0e-15
    assert audit.conservation_enforced is True
    assert audit.cf_compliant is False
    assert audit.metadata["cf_validation_scope"] == "not_evaluated"
    assert audit.metadata["dataset_conformance_checked"] is False
    assert "alias" not in audit.metadata


def test_report_mode_does_not_claim_enforcement_and_intensive_fields_have_no_ledger():
    _, audit = _reduce({"rho": _f64([4.0, 8.0])}, {"rho": "area_density"}, weights=_f64([1.0, 3.0]))
    assert audit.conservation_enforced is False
    assert audit.metadata["conservation_audited_variables"] == ["rho"]
    _, intensive_audit = _reduce({"t": _f64([280.0, 290.0])}, {"t": "intensive"}, enforce=True)
    assert intensive_audit.metadata["conservation_audited_variables"] == []
    assert intensive_audit.conservation_enforced is False
    assert intensive_audit.relative_error == 0.0


def test_fractional_weights_are_normalized_exactly():
    result, _ = _reduce({"t": _f64([[2.0, 4.0]])}, {"t": "intensive"}, weights=_f64([[0.25, 0.25]]))
    assert float(result["t"]) == pytest.approx(3.0)


def test_intensive_means_and_modes_default_to_uniform_weights():
    result, _ = _reduce(
        {"t": _f64([[2.0, 4.0, 6.0]]), "code": _f64([[4.0, 3.0]])},
        {"t": "intensive", "code": "categorical"},
    )
    assert float(result["t"]) == pytest.approx(4.0)
    # A tie goes to the smallest code.
    assert float(result["code"]) == pytest.approx(3.0)


def test_weighted_mode_uses_the_weight_share():
    result, _ = _reduce(
        {"code": _f64([[1.0, 2.0, 2.0]])}, {"code": "categorical"}, weights=_f64([[5.0, 1.0, 1.0]])
    )
    assert float(result["code"]) == pytest.approx(1.0)


def test_an_area_density_needs_its_support_weights():
    with pytest.raises(ValueError, match="weights"):
        _reduce({"rho": _f64([4.0, 8.0])}, {"rho": "area_density"})


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (_f64([[0.0, 0.0]]), "positive weight total"),
        (_f64([0.25, 0.25, 0.5]), "shape"),
        (_f64([[1.0, -1.0]]), "non-negative"),
        (_f64([[1.0, float("nan")]]), "finite"),
    ],
)
def test_invalid_weights_are_rejected(weights, match):
    with pytest.raises(ValueError, match=match):
        _reduce({"t": _f64([[2.0, 4.0]])}, {"t": "intensive"}, weights=weights)


@pytest.mark.parametrize(
    ("basis", "strategy"),
    [
        ("area_density", ScalingStrategy.AREA_WEIGHTED_SUM),
        ("cell_total", ScalingStrategy.AREA_WEIGHTED_MEAN),
        ("intensive", ScalingStrategy.AREA_WEIGHTED_SUM),
        ("intensive", ScalingStrategy.FLOW_CONSERVING),
        ("categorical", ScalingStrategy.AREA_WEIGHTED_MEAN),
        ("intensive", ScalingStrategy.NEAREST),
        ("area_density", ScalingStrategy.BILINEAR),
        ("intensive", ScalingStrategy.MASS_WEIGHTED),
    ],
)
def test_a_caller_strategy_outside_the_basis_raises(basis, strategy):
    with pytest.raises(ValueError, match="conflicts with the declared basis"):
        _reduce({"q": _f64([4.0, 8.0])}, {"q": basis}, weights=_f64([1.0, 3.0]), strategy=strategy)


@pytest.mark.parametrize(
    ("basis", "strategy", "expected"),
    [
        ("area_density", ScalingStrategy.AREA_WEIGHTED_MEAN, 7.0),
        ("area_density", ScalingStrategy.CONSERVATIVE, 7.0),
        ("area_density", ScalingStrategy.FLOW_CONSERVING, 7.0),
        ("cell_total", ScalingStrategy.AREA_WEIGHTED_SUM, 12.0),
        ("cell_total", ScalingStrategy.CONSERVATIVE, 12.0),
        ("intensive", ScalingStrategy.CONSERVATIVE, 7.0),
        ("categorical", ScalingStrategy.MODE, 8.0),
    ],
)
def test_an_allowed_caller_strategy_names_the_basis_law(basis, strategy, expected):
    result, audit = _reduce({"q": _f64([4.0, 8.0])}, {"q": basis}, weights=_f64([1.0, 3.0]), strategy=strategy)
    assert float(result["q"]) == pytest.approx(expected)
    assert audit.strategy is strategy


def test_a_strategy_that_is_not_a_scaling_strategy_raises():
    with pytest.raises(TypeError, match="ScalingStrategy"):
        _reduce({"t": _f64([2.0, 4.0])}, {"t": "intensive"}, strategy="not_a_strategy")


def test_undeclared_non_tensor_and_empty_fields_raise():
    with pytest.raises(ValueError, match="declared spatial basis"):
        _reduce({"t": _f64([2.0, 4.0])}, {})
    with pytest.raises(TypeError, match="tensor"):
        _reduce({"t": [2.0, 4.0]}, {"t": "intensive"})
    with pytest.raises(ValueError, match="no cells"):
        _reduce({"t": torch.zeros(0, dtype=torch.float64)}, {"t": "intensive"})


def test_the_whole_support_kernel_has_no_second_public_entry():
    assert not hasattr(alias_utils, "reduce_spatial_alias_tensor")


@pytest.mark.parametrize(("basis", "values"), [("area_density", [4.0, 8.0]), ("cell_total", [4.0, 8.0])])
def test_the_ledger_is_that_of_the_returned_value(monkeypatch, basis, values):
    """A wrong result can never report a zero residual: the audit integrates the returned value."""
    original = alias_utils._reduce_on_basis

    def doubled(*args, **kwargs):
        return 2.0 * original(*args, **kwargs)

    monkeypatch.setattr(alias_utils, "_reduce_on_basis", doubled)
    with pytest.raises(RuntimeError, match="violates conservation"):
        _reduce({"q": _f64(values)}, {"q": basis}, weights=_f64([1.0, 3.0]), enforce=True)
    with pytest.raises(RuntimeError, match="violates conservation"):
        _reduce(
            {"q": _f64(values)}, {"q": basis}, weights=_f64([1.0, 3.0]), mode=ScalingMode.STRICT_CONSERVATIVE
        )
    result, audit = _reduce({"q": _f64(values)}, {"q": basis}, weights=_f64([1.0, 3.0]))
    expected = 14.0 if basis == "area_density" else 24.0
    assert float(result["q"]) == pytest.approx(expected)
    assert audit.relative_error == pytest.approx(1.0)
    assert audit.metadata["conservation_audit"]["q"].is_conserved is False
    assert audit.conservation_enforced is False
    assert any("not rescaled" in warning for warning in audit.warnings)


def test_float32_rounding_is_not_reported_as_a_conservation_violation(monkeypatch):
    generator = torch.Generator().manual_seed(7)
    values = (torch.rand(64, generator=generator, dtype=torch.float64) * 1.0e-6).to(torch.float32)
    areas = (torch.rand(64, generator=generator, dtype=torch.float64) * 1.0e4 + 1.0).to(torch.float32)
    result, audit = _reduce({"rho": values}, {"rho": "area_density"}, weights=areas, enforce=True)
    assert result["rho"].dtype == torch.float32
    assert audit.metadata["conservation_audit"]["rho"].is_conserved is True
    original = alias_utils._reduce_on_basis
    monkeypatch.setattr(alias_utils, "_reduce_on_basis", lambda *a, **k: 1.5 * original(*a, **k))
    with pytest.raises(RuntimeError, match="violates conservation"):
        _reduce({"rho": values}, {"rho": "area_density"}, weights=areas, enforce=True)


def test_float32_results_keep_the_single_precision_arithmetic():
    """The returned mean is sum(x w) / sum(w) in the field's dtype (bit-for-bit, no float64 detour)."""
    generator = torch.Generator().manual_seed(11)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float32)
    for _ in range(64):
        flux = (torch.rand(2, generator=generator, dtype=torch.float64) * 1.0e-8).to(torch.float32)
        result, _ = _reduce({"f": flux}, {"f": "area_density"}, weights=weights)
        assert torch.equal(result["f"], (flux * weights).sum() / weights.sum())


def test_gradients_flow_through_values_and_weights():
    values = _f64([4.0, 8.0]).requires_grad_(True)
    weights = _f64([1.0, 3.0]).requires_grad_(True)
    result, _ = _reduce({"rho": values}, {"rho": "area_density"}, weights=weights)
    result["rho"].backward()
    torch.testing.assert_close(values.grad, _f64([0.25, 0.75]))
    # d(sum w x / sum w)/dw_i = (x_i - phi) / sum w with phi = 7.
    torch.testing.assert_close(weights.grad, _f64([-0.75, 0.25]))
