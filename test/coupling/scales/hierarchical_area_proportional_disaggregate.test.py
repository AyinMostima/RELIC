"""Owner tests for area-proportional hierarchical disaggregation (ledger D64).

Extensive variables are per-area densities in the hierarchical aggregator (the
conservation enforcer weights them by area), so disaggregating parent j to its
children must satisfy sum_{i in j} X_i A_i = Y_j A_j, i.e.

    X_i = Y_j * A_j / sum_{k in j} A_k

which differs from a plain membership broadcast whenever A_j != sum_k A_k.
Intensive variables are broadcast unchanged.
"""
from __future__ import annotations

import pytest
import torch

from relic.coupling.scales.hierarchical import HierarchicalAggregator, HierarchyIndex


def _index() -> HierarchyIndex:
    topology = {
        "n_source_units": 3,
        "n_target_units": 2,
        "parent_map": {0: 0, 1: 0, 2: 1},
        # Parent 0 is declared with 10 m2 although its children only cover 5 m2.
        "source_areas": [2.0, 3.0, 5.0],
        "target_areas": [10.0, 5.0],
    }
    return HierarchyIndex(source_scale="patch", target_scale="hillslope", topology=topology)


def test_extensive_density_disaggregation_conserves_area_weighted_totals():
    aggregator = HierarchicalAggregator(_index(), weight_type="area", rule="conservative")
    coarse = {"precipitation_flux": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    result, audit = aggregator.disaggregate(coarse, ["precipitation_flux"], mode="conservative")
    fine = result["precipitation_flux"]
    assert torch.allclose(fine, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
    source_areas = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
    target_areas = torch.tensor([10.0, 5.0], dtype=torch.float64)
    assert float((fine * source_areas).sum()) == pytest.approx(float((coarse["precipitation_flux"] * target_areas).sum()))
    assert audit["conservation_errors"]["precipitation_flux"] == pytest.approx(0.0, abs=1e-12)
    assert audit["method_used"]["precipitation_flux"] == "area_proportional"


def test_intensive_disaggregation_is_a_membership_broadcast():
    aggregator = HierarchicalAggregator(_index(), weight_type="area", rule="conservative")
    coarse = {"air_temperature": torch.tensor([285.0, 290.0], dtype=torch.float64)}
    result, _ = aggregator.disaggregate(coarse, ["air_temperature"], mode="conservative")
    assert torch.allclose(result["air_temperature"], torch.tensor([285.0, 285.0, 290.0], dtype=torch.float64))


def _partition_index() -> HierarchyIndex:
    topology = {
        "n_source_units": 3,
        "n_target_units": 2,
        "parent_map": {0: 0, 1: 0, 2: 1},
        "source_areas": [2.0, 3.0, 5.0],
        "target_areas": [5.0, 5.0],
    }
    return HierarchyIndex(source_scale="patch", target_scale="hillslope", topology=topology)


def test_cell_total_basis_is_converted_explicitly_in_both_directions():
    """Per-cell totals are summed per parent and split by member area; the basis is reported."""
    aggregator = HierarchicalAggregator(_partition_index(), weight_type="area", rule="conservative")
    metadata = {"carbon_mass": {"spatial_basis": "cell_total"}}
    totals = {"carbon_mass": torch.tensor([4.0, 6.0, 10.0], dtype=torch.float64)}
    coarse, audit = aggregator.aggregate(totals, ["carbon_mass"], mode="conservative", variable_metadata=metadata)
    assert torch.allclose(coarse["carbon_mass"], torch.tensor([10.0, 10.0], dtype=torch.float64))
    assert audit["quantity_basis"]["carbon_mass"] == "cell_total"
    assert audit["output_basis"]["carbon_mass"] == "cell_total"
    assert audit["conservation_errors"]["carbon_mass"] == pytest.approx(0.0, abs=1e-12)
    fine, down_audit = aggregator.disaggregate(coarse, ["carbon_mass"], mode="conservative", variable_metadata=metadata)
    # Parent 0 (10 kg) is split by member area 2:3; parent 1 keeps its single member.
    assert torch.allclose(fine["carbon_mass"], torch.tensor([4.0, 6.0, 10.0], dtype=torch.float64))
    assert down_audit["output_basis"]["carbon_mass"] == "cell_total"


def test_bare_extensive_metadata_is_ambiguous():
    aggregator = HierarchicalAggregator(_partition_index(), weight_type="area", rule="conservative")
    with pytest.raises(ValueError, match="ambiguous"):
        aggregator.aggregate(
            {"carbon_mass": torch.ones(3, dtype=torch.float64)},
            ["carbon_mass"],
            variable_metadata={"carbon_mass": {"variable_type": "extensive"}},
        )


def test_undeclared_variables_raise_on_every_aggregation_path():
    """No name-pattern inference and no YAML default metadata: an undeclared variable raises."""
    from relic.parameters import get_global_manager

    assert (
        get_global_manager().get("unknown_variable_defaults", namespace="coupling.scales.defaults", required=False)
        is None
    )
    assert not hasattr(HierarchicalAggregator, "_infer_variable_type")
    aggregator = HierarchicalAggregator(_partition_index(), weight_type="area", rule="conservative")
    with pytest.raises(ValueError, match="not declared"):
        aggregator._get_variable_metadata("npp_patch_total")
    fine = {"npp_patch_total": torch.ones(3, dtype=torch.float64)}
    coarse = {"npp_patch_total": torch.ones(2, dtype=torch.float64)}
    for call in (
        lambda: aggregator.aggregate(fine, ["npp_patch_total"]),
        lambda: aggregator.aggregate(fine, ["npp_patch_total"], strategy="area_weighted_mean"),
        lambda: aggregator.disaggregate(coarse, ["npp_patch_total"]),
        lambda: aggregator.upscale(torch.ones(3, dtype=torch.float64)),
        lambda: aggregator.downscale(torch.ones(2, dtype=torch.float64)),
    ):
        with pytest.raises(ValueError, match="not declared"):
            call()
    # The same variable with declared metadata aggregates.
    result, _ = aggregator.aggregate(
        fine, ["npp_patch_total"], variable_metadata={"npp_patch_total": {"spatial_basis": "intensive"}}
    )
    torch.testing.assert_close(result["npp_patch_total"], torch.ones(2, dtype=torch.float64))


def test_child_in_several_parents_is_rejected_unless_weights_sum_to_one():
    index = _partition_index()
    parents = torch.tensor([285.0, 290.0], dtype=torch.float64)
    torch.testing.assert_close(
        index.disaggregate_weights(parents), torch.tensor([285.0, 285.0, 290.0], dtype=torch.float64)
    )
    # Child 0 listed under both parents with full membership: contributions would be summed.
    index.hierarchy_matrix = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="several parents"):
        index.disaggregate_weights(parents)
    # Fractional membership that sums to one is a convex combination.
    index.hierarchy_matrix = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(
        index.disaggregate_weights(parents), torch.tensor([287.5, 285.0, 290.0], dtype=torch.float64)
    )
    # A child without a parent has no disaggregated value.
    index.hierarchy_matrix = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="no parent"):
        index.disaggregate_weights(parents)


def test_extensive_disaggregation_requires_a_strict_partition():
    index = _partition_index()
    index.hierarchy_matrix = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    aggregator = HierarchicalAggregator(index, weight_type="area", rule="conservative")
    with pytest.raises(ValueError, match="partition"):
        aggregator.disaggregate(
            {"precipitation_flux": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            ["precipitation_flux"],
        )


def test_governed_call_runs_the_same_forward():
    aggregator = HierarchicalAggregator(_partition_index(), weight_type="area", rule="conservative")
    out = aggregator(
        {"carbon_mass": torch.tensor([4.0, 6.0, 10.0], dtype=torch.float64)},
        0.0,
        1.0,
        operation="aggregate",
        data={"carbon_mass": torch.tensor([4.0, 6.0, 10.0], dtype=torch.float64)},
        variables=["carbon_mass"],
        variable_metadata={"carbon_mass": {"spatial_basis": "cell_total"}},
    )
    assert torch.allclose(out["carbon_mass_hillslope"], torch.tensor([10.0, 10.0], dtype=torch.float64))


def test_categorical_disaggregation_raises_instead_of_broadcasting_class_codes():
    """A parent class code does not determine the class codes of its children."""
    aggregator = HierarchicalAggregator(_partition_index(), weight_type="area", rule="conservative")
    with pytest.raises(ValueError, match="land_cover.*cannot be disaggregated"):
        aggregator.disaggregate(
            {"land_cover": torch.tensor([2.0, 3.0], dtype=torch.float64)},
            ["land_cover"],
            variable_metadata={"land_cover": {"spatial_basis": "categorical"}},
        )


def test_aggregate_accepts_one_strategy_per_variable():
    """A per-variable strategy mapping aggregates mixed bases in one call."""
    aggregator = HierarchicalAggregator(_partition_index(), weight_type="area", rule="conservative")
    result, audit = aggregator.aggregate(
        {
            "land_cover": torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64),
            "litter_mass": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        },
        ["land_cover", "litter_mass"],
        strategy={"land_cover": "mode", "litter_mass": "flow_conserving"},
        variable_metadata={
            "land_cover": {"spatial_basis": "categorical"},
            "litter_mass": {"spatial_basis": "cell_total"},
        },
    )
    # Parent 0: class 1 on 2 m2, class 2 on 3 m2 -> 2; parent 1: class 4. Cell totals sum.
    torch.testing.assert_close(result["land_cover"].to(torch.float64), torch.tensor([2.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(result["litter_mass"].to(torch.float64), torch.tensor([3.0, 3.0], dtype=torch.float64))
    assert audit["method_used"] == {"land_cover": "mode", "litter_mass": "flow_conserving"}
    with pytest.raises(ValueError, match="no strategy"):
        aggregator.aggregate(
            {"litter_mass": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)},
            ["litter_mass"],
            strategy={"land_cover": "mode"},
            variable_metadata={"litter_mass": {"spatial_basis": "cell_total"}},
        )


def test_hierarchy_level_methods_without_a_declared_basis_are_removed():
    """aggregate_hierarchy/disaggregate_hierarchy could not declare a basis and always raised."""
    from relic.core.abstractions import HierarchicalScalingBase

    for name in ("aggregate_hierarchy", "disaggregate_hierarchy"):
        assert not hasattr(HierarchicalAggregator, name)
        assert name not in HierarchicalScalingBase.__abstractmethods__
