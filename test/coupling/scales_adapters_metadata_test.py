import torch
import pytest

import relic.coupling.scales.adapters as adapters

from relic.coupling.scales.adapters import (
    SpatialAggregation,
    _resolve_variable_basis,
)
from relic.coupling.core.policy import ScalingMode
import relic.coupling.scales.alias_utils as alias_utils
from relic.coupling.scales.alias_utils import aggregate_spatial_alias
from relic.coupling.scales import types as scale_types
from relic.coupling.scales.types import ScalingAudit, ScalingStrategy
from relic.parameters import get_global_manager


def _get_parameter_manager():
    return get_global_manager()


def test_resolve_variable_basis_reads_the_yaml_catalog():
    """The spatial basis comes from the YAML variable catalog when no call metadata is given."""
    pm = _get_parameter_manager()
    basis = scale_types.SpatialQuantityBasis

    flux_basis, source = _resolve_variable_basis("precipitation_flux", pm, None)
    assert flux_basis is basis.AREA_DENSITY
    assert source == "parameter:coupling.scales.defaults"

    temp_basis, source_temp = _resolve_variable_basis("air_temperature", pm, None)
    assert temp_basis is basis.INTENSIVE
    assert source_temp == "parameter:coupling.scales.defaults"

    call_basis, call_source = _resolve_variable_basis(
        "precipitation_flux", pm, {"precipitation_flux": {"spatial_basis": "cell_total"}}
    )
    assert call_basis is basis.CELL_TOTAL
    assert call_source == "call"


def test_resolve_variable_basis_rejects_undeclared_variables():
    """No name heuristic: an undeclared variable raises instead of being classified by name."""
    pm = _get_parameter_manager()
    with pytest.raises(ValueError, match="declare spatial_basis"):
        _resolve_variable_basis("custom_metric_flux", pm, None)


def test_spatial_reduction_follows_the_catalog_basis():
    """SpatialAggregation averages a catalog area density and a catalog intensive field."""
    pm = _get_parameter_manager()
    functor = SpatialAggregation(
        source_scale="src",
        target_scale="tgt",
        parameter_manager=pm,
        preserve_totals=False,
    )
    basis = scale_types.SpatialQuantityBasis
    state = {
        "precipitation_flux": torch.tensor([1.0, 3.0], dtype=torch.float64),
        "air_temperature": torch.tensor([280.0, 290.0], dtype=torch.float64),
    }
    result = functor.forward(state, 0.0, 1.0, weights=torch.tensor([1.0, 3.0], dtype=torch.float64))

    assert float(result["precipitation_flux_aggregated"]) == pytest.approx(2.5)
    assert float(result["air_temperature_aggregated"]) == pytest.approx(287.5)
    audit = functor.last_scaling_audit
    assert audit.metadata["quantity_basis"] == {
        "precipitation_flux": basis.AREA_DENSITY.value,
        "air_temperature": basis.INTENSIVE.value,
    }
    assert audit.metadata["conservation_audited_variables"] == ["precipitation_flux"]
    assert not hasattr(functor, "_resolve_reduction")


def test_temporal_duplicate_adapter_surfaces_are_absent():
    for name in (
        "TemporalAggregation",
        "TemporalScaleCoupling",
        "TemporalAggregator",
        "aggregate_temporal",
        "_map_temporal_method",
    ):
        assert not hasattr(adapters, name)
        assert name not in adapters.__all__


def test_alias_aggregations_never_claim_dataset_cf_conformance():
    data = {"signal": torch.tensor([1.0, 2.0])}
    _, spatial_audit = aggregate_spatial_alias(
        data_dict=data,
        variable_bases={"signal": scale_types.SpatialQuantityBasis.INTENSIVE},
        source_scale="source",
        target_scale="target",
        mode=ScalingMode.BALANCED,
        device=torch.device("cpu"),
        enforce_conservation=False,
        tolerance=1.0e-8,
        strategy=ScalingStrategy.AREA_WEIGHTED_MEAN,
    )

    assert spatial_audit.cf_compliant is False
    assert spatial_audit.metadata["cf_validation_scope"] == "not_evaluated"
    assert spatial_audit.metadata["dataset_conformance_checked"] is False
    # The area-free conservative alias sum is gone: conservative alias scale pairs run through
    # the hierarchical path (declared topology and areas, basis-aware kernels, ledger audit).
    assert not hasattr(alias_utils, "aggregate_conservative_alias")


def test_spatial_quantity_basis_is_one_explicit_enum():
    """One canonical spatial representation enum; a bare 'extensive' label is ambiguous."""
    basis = scale_types.SpatialQuantityBasis
    assert {member.value for member in basis} == {
        "intensive",
        "cell_total",
        "area_density",
        "categorical",
    }
    assert basis.parse("area_density") is basis.AREA_DENSITY
    assert basis.parse(basis.CELL_TOTAL) is basis.CELL_TOTAL
    assert basis.AREA_DENSITY.is_extensive and basis.CELL_TOTAL.is_extensive
    assert not basis.INTENSIVE.is_extensive and not basis.CATEGORICAL.is_extensive
    with pytest.raises(ValueError, match="ambiguous"):
        basis.parse("extensive")
    with pytest.raises(ValueError, match="Unknown spatial quantity basis"):
        basis.parse("density")


def test_spatial_basis_owner_is_exported_from_the_scales_package():
    """The package re-exports the single owner objects, not copies."""
    import relic.coupling.scales as scales

    assert scales.SpatialQuantityBasis is scale_types.SpatialQuantityBasis
    assert scales.resolve_spatial_basis is scale_types.resolve_spatial_basis
    assert scales.convert_spatial_basis is scale_types.convert_spatial_basis
    for name in ("SpatialQuantityBasis", "resolve_spatial_basis", "convert_spatial_basis"):
        assert name in scales.__all__


def test_resolve_spatial_basis_requires_explicit_extensive_representation():
    """Metadata resolution: intensive/categorical map directly, extensive needs spatial_basis."""
    resolve = scale_types.resolve_spatial_basis
    basis = scale_types.SpatialQuantityBasis
    assert resolve("x", {"spatial_basis": "cell_total"}) is basis.CELL_TOTAL
    assert (
        resolve("x", {"variable_type": "extensive", "spatial_basis": "area_density"})
        is basis.AREA_DENSITY
    )
    assert resolve("t", {"variable_type": "intensive"}) is basis.INTENSIVE
    assert resolve("c", {"variable_type": "categorical"}) is basis.CATEGORICAL
    with pytest.raises(ValueError, match="ambiguous"):
        resolve("x", {"variable_type": "extensive"})
    with pytest.raises(ValueError, match="inconsistent"):
        resolve("x", {"variable_type": "intensive", "spatial_basis": "cell_total"})
    with pytest.raises(ValueError, match="declare"):
        resolve("x", None)
    with pytest.raises(ValueError, match="declare"):
        resolve("x", {"units": "kg"})


def test_convert_spatial_basis_between_cell_total_and_area_density():
    """X_i = x_i A_i and x_i = X_i / A_i, differentiable, no conversion for intensive fields."""
    basis = scale_types.SpatialQuantityBasis
    convert = scale_types.convert_spatial_basis
    totals = torch.tensor([4.0, 8.0], dtype=torch.float64)  # kg per cell
    areas = torch.tensor([1.0, 2.0], dtype=torch.float64)  # m2
    density = convert(totals, areas, basis.CELL_TOTAL, basis.AREA_DENSITY)
    assert torch.equal(density, torch.tensor([4.0, 4.0], dtype=torch.float64))
    assert torch.equal(convert(density, areas, basis.AREA_DENSITY, basis.CELL_TOTAL), totals)
    assert convert(totals, areas, basis.CELL_TOTAL, basis.CELL_TOTAL) is totals
    batch = convert(torch.stack([totals, 2.0 * totals]), areas, basis.CELL_TOTAL, basis.AREA_DENSITY)
    assert torch.equal(batch[1], torch.tensor([8.0, 8.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="intensive"):
        convert(totals, areas, basis.INTENSIVE, basis.AREA_DENSITY)
    with pytest.raises(ValueError, match="positive"):
        convert(totals, torch.tensor([1.0, 0.0], dtype=torch.float64), basis.CELL_TOTAL, basis.AREA_DENSITY)
    with pytest.raises(ValueError, match="broadcast"):
        convert(totals, torch.ones(3, dtype=torch.float64), basis.CELL_TOTAL, basis.AREA_DENSITY)
    x = totals.clone().requires_grad_(True)
    convert(x, areas, basis.CELL_TOTAL, basis.AREA_DENSITY).sum().backward()
    assert torch.allclose(x.grad, 1.0 / areas)


def test_scale_variable_catalog_declares_explicit_spatial_basis():
    """Every catalog entry resolves without ambiguity; per-area extensive entries are area densities."""
    catalog = _get_parameter_manager().get("coupling.scales.defaults.variables", required=True)
    basis = scale_types.SpatialQuantityBasis
    extensive_entries = 0
    for name, meta in catalog.items():
        if name.startswith("__"):
            continue
        resolved = scale_types.resolve_spatial_basis(name, meta)
        if meta.get("variable_type") == "extensive":
            extensive_entries += 1
            assert resolved is basis.AREA_DENSITY, name
            assert "m-2" in meta["units"] or "m-3" in meta["units"], name
    assert extensive_entries > 0


def test_scaling_audit_rejects_positive_dataset_cf_claim():
    with pytest.raises(ValueError, match="dataset-level CF conformance"):
        ScalingAudit(
            operation="aggregate",
            source_scale="source",
            target_scale="target",
            variables=["signal"],
            mode=ScalingMode.BALANCED,
            strategy=ScalingStrategy.AREA_WEIGHTED_MEAN,
            mass_balance_error=0.0,
            relative_error=0.0,
            cf_compliant=True,
            conservation_enforced=False,
            physical_constraints_applied=False,
            warnings=[],
            metadata={},
        )
