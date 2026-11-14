"""
HierarchicalAggregator L1 Unit Tests

Exercises the real hierarchical aggregation pipeline (no mocks) to verify that
flux conservation, intensive averaging, functor interface behaviour, and
diagnostic outputs all behave as expected.

Scientific context:
    - Extensive variable aggregation preserves integrated fluxes.
    - Intensive variable aggregation produces area-weighted means.
    - Disaggregation redistributes intensive variables proportionally to area.

Mathematical expressions:
    Y_j = sum_i X_i * A_i / A_j
    phi_j = sum_i phi_i * A_i / sum_i A_i
    eps = |sum_j Y_j * A_j - sum_i X_i * A_i| / sum_i X_i * A_i
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import pytest
import torch

# Ensure relic package is importable in the pytest environment
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from relic.coupling.scales.hierarchical.aggregator import HierarchicalAggregator
from relic.coupling.scales.hierarchical.hierarchy_index import HierarchyIndex


@pytest.fixture
def hierarchy_index() -> HierarchyIndex:
    """Default 6 patch -> 2 hillslope hierarchy matching production defaults."""
    return HierarchyIndex(
        source_scale="patch",
        target_scale="hillslope",
        n_source_units=6,
        device=torch.device("cpu"),
    )


@pytest.fixture
def aggregator(hierarchy_index: HierarchyIndex) -> HierarchicalAggregator:
    """Instantiate the production aggregator on CPU."""
    return HierarchicalAggregator(
        index=hierarchy_index,
        device=torch.device("cpu"),
        weight_type="area",
        rule="conservative",
    )


def _group_mean(values: torch.Tensor, group_size: int) -> torch.Tensor:
    """Utility to compute reference means for verification."""
    return values.view(-1, group_size).mean(dim=1)


def _topology_aggregator(parent_map, source_areas, target_areas) -> HierarchicalAggregator:
    index = HierarchyIndex(
        source_scale="patch",
        target_scale="hillslope",
        device=torch.device("cpu"),
        topology={
            "n_source_units": len(source_areas),
            "n_target_units": len(target_areas),
            "parent_map": parent_map,
            "source_areas": source_areas,
            "target_areas": target_areas,
        },
    )
    return HierarchicalAggregator(index=index, device=torch.device("cpu"), weight_type="area", rule="conservative")


DENSITY = {"carbon_density": {"spatial_basis": "area_density"}}


def _bilinear_square_aggregator(scale: float) -> HierarchicalAggregator:
    """Four nodes of a unit square (coordinates multiplied by ``scale``) and one target at its centre."""
    source = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    index = HierarchyIndex(
        source_scale="patch",
        target_scale="hillslope",
        device=torch.device("cpu"),
        topology={
            "n_source_units": 4,
            "n_target_units": 1,
            "parent_map": {0: 0, 1: 0, 2: 0, 3: 0},
            "source_areas": [1.0, 1.0, 1.0, 1.0],
            "target_areas": [4.0],
            "source_coordinates": [[scale * x, scale * y] for x, y in source],
            "target_coordinates": [[scale * 0.5, scale * 0.5]],
        },
    )
    return HierarchicalAggregator(index=index, device=torch.device("cpu"), weight_type="area", rule="conservative")


class TestNumericalEpsilon:
    """The aggregator reads one key, coupling.scales.defaults.epsilon; nothing falls back or is swallowed."""

    def test_the_epsilon_is_the_scale_default_parameter(self, aggregator: HierarchicalAggregator) -> None:
        from relic.parameters import get_global_manager

        expected = get_global_manager().get("epsilon", namespace="coupling.scales.defaults", required=True)
        assert aggregator.epsilon == pytest.approx(float(expected), rel=1.0e-6)

    @pytest.mark.parametrize(
        "error", [KeyError("Required parameter 'coupling.scales.defaults.epsilon' not found"),
                  RuntimeError("coupling.scales.defaults.epsilon: parameter database unavailable")]
    )
    def test_a_missing_epsilon_raises_instead_of_falling_back(self, monkeypatch, error) -> None:
        original = HierarchicalAggregator.get_parameter_value

        def without_scale_epsilon(self, name, *args, **kwargs):
            if name == "epsilon" and kwargs.get("namespace") == "coupling.scales.defaults":
                raise error
            return original(self, name, *args, **kwargs)

        monkeypatch.setattr(HierarchicalAggregator, "get_parameter_value", without_scale_epsilon)
        with pytest.raises(type(error), match="coupling.scales.defaults.epsilon"):
            HierarchicalAggregator(
                index=HierarchyIndex(source_scale="patch", target_scale="hillslope", n_source_units=6,
                                     device=torch.device("cpu")),
                device=torch.device("cpu"),
                weight_type="area",
                rule="conservative",
            )

    @pytest.mark.parametrize("scale", [1.0, 1.0e3, 1.0e-16])
    def test_bilinear_weights_do_not_depend_on_the_coordinate_unit(self, scale) -> None:
        # Multiplying every coordinate by c > 0 leaves tx and ty unchanged: the centre of the
        # square is the mean of its four corners, 2.5, in any coordinate unit.
        aggregator = _bilinear_square_aggregator(scale)
        result, audit = aggregator.aggregate(
            data={"surface_temperature": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)},
            variables=["surface_temperature"],
            strategy="bilinear",
            variable_metadata={"surface_temperature": {"spatial_basis": "intensive", "units": "K"}},
        )
        assert audit["method_used"]["surface_temperature"] == "bilinear"
        torch.testing.assert_close(result["surface_temperature"], torch.tensor([2.5], dtype=torch.float64))
TOTALS = {"carbon_mass": {"spatial_basis": "cell_total"}}


class TestConservativeKernelsArePartitionBased:
    """Documented conservative definitions: per-parent ledgers, no hidden global rescale."""

    def test_orphan_amount_is_reported_not_spread_into_parents(self) -> None:
        aggregator = _topology_aggregator({0: 0, 1: 1}, [1.0, 1.0, 1.0], [1.0, 1.0])
        data = {"carbon_density": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)}
        result, audit = aggregator.aggregate(data, ["carbon_density"], strategy="conservative", variable_metadata=DENSITY)
        assert torch.allclose(result["carbon_density"], torch.tensor([1.0, 1.0], dtype=torch.float64))
        assert audit["uncovered_source_amount"]["carbon_density"] == pytest.approx(1.0)
        assert audit["orphan_source_units"] == [2]
        # The covered-support ledger closes exactly.
        assert audit["conservation_errors"]["carbon_density"] == pytest.approx(0.0, abs=1e-12)

    def test_fractional_membership_splits_the_child_amount(self) -> None:
        aggregator = _topology_aggregator({0: 0, 1: 0, 2: 1}, [1.0, 1.0, 1.0], [1.5, 1.5])
        aggregator.index.hierarchy_matrix = torch.tensor(
            [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
        )
        data = {"carbon_density": torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64)}
        result, _ = aggregator.aggregate(data, ["carbon_density"], strategy="conservative", variable_metadata=DENSITY)
        parent_amounts = result["carbon_density"] * torch.tensor([1.5, 1.5], dtype=torch.float64)
        # 2*0.5 + 1 = 2 and 2*0.5 + 3 = 4 (the former binary count plus global rescale gave 2.25/3.75)
        assert torch.allclose(parent_amounts, torch.tensor([2.0, 4.0], dtype=torch.float64))

    def test_child_in_several_parents_without_unit_weights_is_rejected(self) -> None:
        aggregator = _topology_aggregator({0: 0, 1: 0, 2: 1}, [1.0, 1.0, 1.0], [1.5, 1.5])
        aggregator.index.hierarchy_matrix = torch.tensor(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
        )
        data = {"carbon_density": torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64)}
        with pytest.raises(RuntimeError, match="several parents"):
            aggregator.aggregate(data, ["carbon_density"], strategy="conservative", variable_metadata=DENSITY)

    def test_parent_with_members_needs_a_positive_area(self) -> None:
        aggregator = _topology_aggregator({0: 0, 1: 1}, [1.0, 1.0], [1.0, 0.0])
        data = {"carbon_density": torch.tensor([1.0, 1.0], dtype=torch.float64)}
        with pytest.raises(RuntimeError, match="positive"):
            aggregator.aggregate(data, ["carbon_density"], strategy="conservative", variable_metadata=DENSITY)

    def test_area_weighted_sum_sums_member_totals_on_a_column_normalised_index(
        self, aggregator: HierarchicalAggregator
    ) -> None:
        # The parameter-built index stores column-normalised weights 1/3 per member.
        assert torch.allclose(
            aggregator.index.get_hierarchy_matrix().sum(dim=0).to(torch.float64),
            torch.ones(2, dtype=torch.float64),
        )
        data = {"carbon_mass": torch.arange(1.0, 7.0, dtype=torch.float64)}
        result, audit = aggregator.aggregate(data, ["carbon_mass"], strategy="area_weighted_sum", variable_metadata=TOTALS)
        assert torch.allclose(result["carbon_mass"].to(torch.float64), torch.tensor([6.0, 15.0], dtype=torch.float64))
        assert audit["conservation_errors"]["carbon_mass"] == pytest.approx(0.0, abs=1e-12)

    def test_amount_kernels_reject_intensive_fields(self, aggregator: HierarchicalAggregator) -> None:
        data = {"canopy_temperature": torch.full((6,), 290.0, dtype=torch.float64)}
        metadata = {"canopy_temperature": {"spatial_basis": "intensive"}}
        with pytest.raises(RuntimeError, match="amounts"):
            aggregator.aggregate(data, ["canopy_temperature"], strategy="area_weighted_sum", variable_metadata=metadata)
        # The balanced default picks the area-weighted mean for an intensive field.
        result, audit = aggregator.aggregate(data, ["canopy_temperature"], mode="balanced", variable_metadata=metadata)
        assert audit["method_used"]["canopy_temperature"] == "area_weighted_mean"
        assert torch.allclose(result["canopy_temperature"].to(torch.float64), torch.full((2,), 290.0, dtype=torch.float64))


_ROUGHNESS_METADATA = {"surface_roughness": {"spatial_basis": "intensive", "units": "m"}}


def _roughness(aggregator: HierarchicalAggregator, z0: torch.Tensor) -> torch.Tensor:
    result, _audit = aggregator.aggregate(
        data={"surface_roughness": z0},
        variables=["surface_roughness"],
        strategy="effective_roughness_length",
        variable_metadata=_ROUGHNESS_METADATA,
    )
    return result["surface_roughness"]


class TestEffectiveRoughnessLength:
    """z0_eff = z_r exp(-(sum_j f_j / ln(z_r/z0_j)^2)^(-1/2)) (Byun et al. 1999, Eq. 12-37')."""

    def test_blending_height_is_a_yaml_parameter(self) -> None:
        from relic.parameters import get_global_manager

        spec = get_global_manager().get_parameter("coupling.scales.defaults.roughness_blending_height")
        assert float(spec.value) == pytest.approx(10.0)
        assert spec.units == "m"

    def test_one_class_returns_its_roughness_length(self) -> None:
        agg = _topology_aggregator({0: 0, 1: 0, 2: 1}, [1.0, 3.0, 2.0], [4.0, 2.0])
        z0 = torch.tensor([0.05, 0.05, 0.8], dtype=torch.float64)
        torch.testing.assert_close(
            _roughness(agg, z0), torch.tensor([0.05, 0.8], dtype=torch.float64), rtol=1e-12, atol=0.0
        )

    def test_two_classes_match_the_closed_form(self) -> None:
        import math

        z_r = 10.0
        agg = _topology_aggregator({0: 0, 1: 0}, [2.0, 2.0], [4.0])
        z0 = torch.tensor([0.01, 1.0], dtype=torch.float64)
        inverse_sq = 0.5 / math.log(z_r / 0.01) ** 2 + 0.5 / math.log(z_r / 1.0) ** 2
        expected = z_r * math.exp(-(inverse_sq ** -0.5))
        assert expected == pytest.approx(0.45541, abs=5e-5)
        torch.testing.assert_close(
            _roughness(agg, z0), torch.tensor([expected], dtype=torch.float64), rtol=1e-12, atol=0.0
        )
        # Unequal area fractions weight the drag coefficients, not the lengths.
        agg_unequal = _topology_aggregator({0: 0, 1: 0}, [3.0, 1.0], [4.0])
        inverse_sq = 0.75 / math.log(z_r / 0.01) ** 2 + 0.25 / math.log(z_r / 1.0) ** 2
        torch.testing.assert_close(
            _roughness(agg_unequal, z0),
            torch.tensor([z_r * math.exp(-(inverse_sq ** -0.5))], dtype=torch.float64),
            rtol=1e-12,
            atol=0.0,
        )

    def test_gradient_is_finite(self) -> None:
        agg = _topology_aggregator({0: 0, 1: 0}, [2.0, 2.0], [4.0])
        z0 = torch.tensor([0.01, 1.0], dtype=torch.float64, requires_grad=True)
        _roughness(agg, z0).sum().backward()
        assert z0.grad is not None
        assert bool(torch.isfinite(z0.grad).all())
        assert bool((z0.grad > 0).all())

    @pytest.mark.parametrize("bad", [0.0, -0.1, 10.0, 12.0])
    def test_roughness_outside_the_open_interval_raises(self, bad: float) -> None:
        agg = _topology_aggregator({0: 0, 1: 0}, [2.0, 2.0], [4.0])
        with pytest.raises(RuntimeError, match="0 < z0 < z_r"):
            _roughness(agg, torch.tensor([0.1, bad], dtype=torch.float64))

    def test_builder_and_landscape_model_use_the_method(self) -> None:
        from relic.coupling.scales.hierarchical.model_builder import AggregationMethod
        from relic.models import landscape

        assert AggregationMethod.EFFECTIVE_ROUGHNESS_LENGTH.value == "effective_roughness_length"
        assert (
            landscape.EXPOSED_VARIABLES["surface_roughness"]["aggregation_method"]
            == "effective_roughness_length"
        )


_CATEGORICAL_METADATA = {"land_cover": {"spatial_basis": "categorical", "units": "1"}}


def _land_cover_mode(aggregator: HierarchicalAggregator, codes: torch.Tensor, strategy="mode", mode="conservative"):
    result, audit = aggregator.aggregate(
        data={"land_cover": codes},
        variables=["land_cover"],
        strategy=strategy,
        mode=mode,
        variable_metadata=_CATEGORICAL_METADATA,
    )
    return result["land_cover"], audit


class TestAreaWeightedMode:
    """mode_j = argmax_c sum_{i: x_i = c} A_i M_ij; ties -> smallest class code."""

    def test_dominant_class_is_the_largest_area_share_not_the_most_cells(self) -> None:
        # Parent 0: classes [1, 2, 2] on areas [5, 1, 1] -> class 1 (5 m2) beats class 2 (2 cells, 2 m2).
        # Parent 1: classes [4, 3] on equal areas -> tie broken to the smaller code 3.
        agg = _topology_aggregator({0: 0, 1: 0, 2: 0, 3: 1, 4: 1}, [5.0, 1.0, 1.0, 1.0, 1.0], [7.0, 2.0])
        codes = torch.tensor([1.0, 2.0, 2.0, 4.0, 3.0], dtype=torch.float64)
        result, audit = _land_cover_mode(agg, codes)
        torch.testing.assert_close(result, torch.tensor([1.0, 3.0], dtype=torch.float64))
        assert audit["method_used"]["land_cover"] == "mode"

    def test_mode_is_the_builder_method_and_the_only_mode_variant(self) -> None:
        from relic.coupling.scales.hierarchical.model_builder import AggregationMethod

        agg = _topology_aggregator({0: 0, 1: 0}, [1.0, 1.0], [2.0])
        assert AggregationMethod.MODE.value in agg.method_map
        assert "area_weighted_mode" not in agg.method_map
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            _land_cover_mode(agg, torch.tensor([1.0, 2.0], dtype=torch.float64), strategy="area_weighted_mode")

    def test_categorical_fields_auto_select_the_mode(self) -> None:
        agg = _topology_aggregator({0: 0, 1: 0, 2: 0}, [1.0, 1.0, 4.0], [6.0])
        codes = torch.tensor([2.0, 2.0, 7.0], dtype=torch.float64)
        for mode in ("conservative", "balanced", "fast_approx"):
            result, audit = _land_cover_mode(agg, codes, strategy=None, mode=mode)
            torch.testing.assert_close(result, torch.tensor([7.0], dtype=torch.float64))
            assert audit["method_used"]["land_cover"] == "mode"

    def test_shared_child_counts_with_its_membership_share(self) -> None:
        index = HierarchyIndex(
            source_scale="patch",
            target_scale="hillslope",
            device=torch.device("cpu"),
            topology={
                "n_source_units": 3,
                "n_target_units": 2,
                "parent_map": {0: 0, 2: 1},
                "source_areas": [1.0, 3.0, 2.5],
                "target_areas": [2.5, 4.0],
            },
        )
        index.hierarchy_matrix[1, 0] = 0.5
        index.hierarchy_matrix[1, 1] = 0.5
        agg = HierarchicalAggregator(index=index, device=torch.device("cpu"), weight_type="area", rule="conservative")
        # Parent 0: class 1 on 1 m2, class 2 on 0.5 * 3 = 1.5 m2 -> 2.
        # Parent 1: class 2 on 1.5 m2, class 3 on 2.5 m2 -> 3.
        result, _audit = _land_cover_mode(agg, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        torch.testing.assert_close(result, torch.tensor([2.0, 3.0], dtype=torch.float64))

    @pytest.mark.parametrize(
        "strategy",
        ["area_weighted_mean", "arithmetic_mean", "harmonic_mean", "geometric_mean", "bilinear",
         "conservative", "bounded_area_mean", "circular_mean"],
    )
    def test_value_methods_are_rejected_for_categorical_fields(self, strategy) -> None:
        agg = _topology_aggregator({0: 0, 1: 0}, [1.0, 1.0], [2.0])
        with pytest.raises(RuntimeError, match="categorical"):
            _land_cover_mode(agg, torch.tensor([1.0, 2.0], dtype=torch.float64), strategy=strategy)

    def test_unknown_strategies_are_rejected(self) -> None:
        agg = _topology_aggregator({0: 0, 1: 0}, [1.0, 1.0], [2.0])
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            agg.aggregate(
                data={"x": torch.tensor([1.0, 2.0], dtype=torch.float64)},
                variables=["x"],
                strategy="not_a_method",
                variable_metadata={"x": {"spatial_basis": "intensive"}},
            )


def _shared_child_orphan_aggregator() -> HierarchicalAggregator:
    """Children 0..3 with areas [1, 2, 3, 4] m2: child 1 shared 0.5/0.5, child 3 an orphan.

    Area fractions f_ij = A_i M_ij / sum_k A_k M_kj: parent 0 -> [0.5, 0.5] over children 0, 1;
    parent 1 -> [0.25, 0.75] over children 1, 2.
    """
    index = HierarchyIndex(
        source_scale="patch",
        target_scale="hillslope",
        device=torch.device("cpu"),
        topology={
            "n_source_units": 4,
            "n_target_units": 2,
            "parent_map": {0: 0, 2: 1},
            "source_areas": [1.0, 2.0, 3.0, 4.0],
            "target_areas": [2.0, 4.0],
        },
    )
    index.hierarchy_matrix[1, 0] = 0.5
    index.hierarchy_matrix[1, 1] = 0.5
    return HierarchicalAggregator(index=index, device=torch.device("cpu"), weight_type="area", rule="conservative")


def _intensive(aggregator, values, strategy):
    result, _audit = aggregator.aggregate(
        data={"x": torch.tensor(values, dtype=torch.float64)},
        variables=["x"],
        strategy=strategy,
        variable_metadata={"x": {"spatial_basis": "intensive"}},
    )
    return result["x"]


class TestValueKernelsUseTheMembershipMapping:
    """Means over f_ij from HierarchyIndex.membership_mapping; the orphan value never enters."""

    def test_area_weighted_and_bounded_means(self) -> None:
        agg = _shared_child_orphan_aggregator()
        torch.testing.assert_close(
            _intensive(agg, [2.0, 8.0, 4.0, 100.0], "area_weighted_mean"),
            torch.tensor([5.0, 5.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            _intensive(agg, [0.2, 0.8, 0.4, 0.9], "bounded_area_mean"),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        )
        with pytest.raises(RuntimeError, match=r"\[0, 1\]"):
            _intensive(agg, [0.2, 1.8, 0.4, 0.9], "bounded_area_mean")

    def test_harmonic_and_geometric_means(self) -> None:
        import math

        agg = _shared_child_orphan_aggregator()
        harmonic = [1.0 / (0.5 / 2.0 + 0.5 / 8.0), 1.0 / (0.25 / 8.0 + 0.75 / 4.0)]
        geometric = [math.exp(0.5 * math.log(2.0) + 0.5 * math.log(8.0)),
                     math.exp(0.25 * math.log(8.0) + 0.75 * math.log(4.0))]
        torch.testing.assert_close(
            _intensive(agg, [2.0, 8.0, 4.0, 100.0], "harmonic_mean"),
            torch.tensor(harmonic, dtype=torch.float64),
        )
        torch.testing.assert_close(
            _intensive(agg, [2.0, 8.0, 4.0, 100.0], "geometric_mean"),
            torch.tensor(geometric, dtype=torch.float64),
        )
        for strategy in ("harmonic_mean", "geometric_mean"):
            with pytest.raises(RuntimeError, match="positive"):
                _intensive(agg, [2.0, 0.0, 4.0, 100.0], strategy)

    def test_circular_mean(self) -> None:
        import math

        agg = _shared_child_orphan_aggregator()
        angles = [0.0, math.pi / 2.0, math.pi / 2.0, math.pi]
        expected = [math.atan2(0.5, 0.5), math.atan2(1.0, 0.0)]
        torch.testing.assert_close(
            _intensive(agg, angles, "circular_mean"), torch.tensor(expected, dtype=torch.float64)
        )

    def test_invalid_shared_weights_and_memberless_parents_raise(self) -> None:
        agg = _shared_child_orphan_aggregator()
        agg.index.hierarchy_matrix[1, 1] = 0.7  # shares of child 1 no longer sum to one
        with pytest.raises(RuntimeError, match="sum to one"):
            _intensive(agg, [2.0, 8.0, 4.0, 100.0], "geometric_mean")
        empty = _topology_aggregator({0: 0, 1: 0}, [1.0, 1.0], [2.0, 1.0])
        with pytest.raises(RuntimeError, match="no member area"):
            _intensive(empty, [1.0, 2.0], "harmonic_mean")

    def test_value_kernels_have_no_raw_matrix_or_name_branches(self) -> None:
        import inspect

        for name in ("_area_weighted_mean", "_bounded_area_mean", "_harmonic_mean",
                     "_geometric_mean", "_circular_mean", "_component_wise_mean"):
            source = inspect.getsource(getattr(HierarchicalAggregator, name))
            assert "get_hierarchy_matrix" not in source, name
            assert "stomatal_resistance" not in source, name
            assert "torch.clamp" not in source, name


class TestHierarchicalAggregatorL1:
    """L1 tests for hierarchical aggregation and disaggregation."""

    def test_initialization_metadata(self, aggregator: HierarchicalAggregator, hierarchy_index: HierarchyIndex) -> None:
        """Aggregator exposes correct scales, rules, and metadata."""
        assert aggregator.index is hierarchy_index
        assert aggregator.source_scale == "patch"
        assert aggregator.target_scale == "hillslope"
        assert aggregator.weight_type == "area"
        assert aggregator.rule == "conservative"
        assert "conservative" in aggregator.method_map
        assert "area_weighted_mean" in aggregator.method_map
        assert aggregator.metadata.name == "HierarchicalAggregator_patch_to_hillslope"

    def test_initialization_requires_declared_hierarchy(self) -> None:
        """Missing index and topology fails instead of inventing a hierarchy."""
        with pytest.raises(ValueError, match="index or topology"):
            HierarchicalAggregator(device=torch.device("cpu"), weight_type="area", rule="conservative")

    def test_initialization_accepts_explicit_topology(self) -> None:
        """Explicit topology is routed through the canonical hierarchy index."""
        topology = {
            "n_source_units": 2,
            "n_target_units": 1,
            "parent_map": {0: 0, 1: 0},
            "source_areas": [2.0, 3.0],
            "target_areas": [5.0],
        }

        explicit = HierarchicalAggregator(
            hierarchy_levels=["patch", "hillslope"],
            topology=topology,
            device=torch.device("cpu"),
            weight_type="area",
            rule="conservative",
        )

        assert isinstance(explicit.index, HierarchyIndex)
        assert explicit.index.n_source_units == 2
        assert explicit.index.n_target_units == 1

    def test_aggregate_extensive_variable_conservation(self, aggregator: HierarchicalAggregator, hierarchy_index: HierarchyIndex) -> None:
        """Extensive variables maintain their integrated totals after aggregation."""
        # net_primary_productivity: a catalogued area density (kg C m-2 s-1 per unit ground area).
        data = {"net_primary_productivity": torch.arange(1.0, 7.0) * 1.0e-8}
        result, audit = aggregator.aggregate(data=data, variables=["net_primary_productivity"], strategy="conservative")

        aggregated = result["net_primary_productivity"]
        assert aggregated.shape[0] == hierarchy_index.n_target_units
        assert audit["method_used"]["net_primary_productivity"] == "conservative"

        source_total = (data["net_primary_productivity"] * hierarchy_index.get_source_areas()).sum()
        target_total = (aggregated * hierarchy_index.get_target_areas()).sum()
        rel_error = torch.abs(target_total - source_total) / torch.abs(source_total)
        assert rel_error < 1.0e-8

    def test_aggregate_intensive_variable_area_weighted(self, aggregator: HierarchicalAggregator) -> None:
        """Intensive variables return area-weighted means across groups."""
        data = {"air_temperature": torch.tensor([290.0, 291.0, 292.0, 293.0, 294.0, 295.0])}
        result, audit = aggregator.aggregate(data=data, variables=["air_temperature"], strategy="area_weighted_mean")

        aggregated = result["air_temperature"]
        assert audit["method_used"]["air_temperature"] == "area_weighted_mean"
        expected = _group_mean(data["air_temperature"], group_size=3)
        assert torch.allclose(aggregated, expected, atol=1.0e-8)

    def test_disaggregate_extensive_variable(self, aggregator: HierarchicalAggregator, hierarchy_index: HierarchyIndex) -> None:
        """Disaggregation of extensive variables distributes totals according to area."""
        coarse = {"net_primary_productivity": torch.tensor([6.0e-8, 1.5e-7])}
        result, audit = aggregator.disaggregate(data=coarse, variables=["net_primary_productivity"])

        fine = result["net_primary_productivity"]
        assert fine.shape[0] == hierarchy_index.n_source_units
        assert audit["method_used"]["net_primary_productivity"] == "area_proportional"

        source_total = (coarse["net_primary_productivity"] * hierarchy_index.get_target_areas()).sum()
        target_total = (fine * hierarchy_index.get_source_areas()).sum()
        rel_error = torch.abs(target_total - source_total) / torch.abs(source_total)
        assert rel_error < 1.0e-8

    def test_disaggregate_intensive_variable(self, aggregator: HierarchicalAggregator) -> None:
        """Intensive variables replicate the parent value across child units."""
        coarse = {"air_temperature": torch.tensor([291.0, 294.0])}
        result, audit = aggregator.disaggregate(data=coarse, variables=["air_temperature"])

        fine = result["air_temperature"]
        assert audit["method_used"]["air_temperature"] == "area_proportional"
        expected = torch.tensor([291.0, 291.0, 291.0, 294.0, 294.0, 294.0])
        assert torch.allclose(fine, expected, atol=1.0e-8)

    def test_forward_tensor_input(self, aggregator: HierarchicalAggregator) -> None:
        """Forward with tensor input returns hillslope-labelled state update."""
        state = {
            "operation": "aggregate",
            "data": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "strategy": "conservative",
            # A tensor input is the variable "default_var"; its basis is declared explicitly.
            "variable_metadata": {"default_var": {"spatial_basis": "intensive"}},
        }
        updates = aggregator.forward(state, time=0.0, dt=1800.0)
        # Equal patch areas: the intensive conservative aggregate is the group mean.
        assert torch.allclose(
            updates["hillslope_data"].to(torch.float64), torch.tensor([2.0, 5.0], dtype=torch.float64)
        )
        diagnostics = updates["_hierarchical_diagnostics"]
        assert diagnostics["quantity_basis"]["default_var"] == "intensive"
        assert diagnostics["operation"] == "aggregate"
        assert diagnostics["source_scale"] == "patch"
        assert diagnostics["target_scale"] == "hillslope"

    def test_forward_auto_detects_operation(self, aggregator: HierarchicalAggregator) -> None:
        """When no operation provided, aggregator infers aggregate/disaggregate."""
        state = {
            "data": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "variable_metadata": {"default_var": {"spatial_basis": "intensive"}},
        }
        updates = aggregator.forward(state, time=0.0, dt=1800.0)
        assert updates["_hierarchical_diagnostics"]["operation"] == "aggregate"
        assert updates["hillslope_data"].shape == (2,)

    def test_forward_rejects_named_variables_for_a_tensor_input(self, aggregator: HierarchicalAggregator) -> None:
        """A tensor input is 'default_var'; other names raise instead of echoing the input back."""
        state = {
            "data": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "variables": ["air_temperature"],
        }
        with pytest.raises(ValueError, match="default_var"):
            aggregator.forward(state, time=0.0, dt=1800.0)

    def test_forward_multiple_variables_dict(self, aggregator: HierarchicalAggregator) -> None:
        """Forward handles dict input and labels outputs with target scale suffix."""
        state = {
            "operation": "aggregate",
            "variables": ["air_temperature", "moisture"],
            "data": {
                "air_temperature": torch.tensor([290.0, 291.0, 292.0, 293.0, 294.0, 295.0]),
                "moisture": torch.tensor([0.1, 0.2, 0.15, 0.3, 0.25, 0.2]),
            },
            "variable_metadata": {"moisture": {"spatial_basis": "intensive"}},
        }
        updates = aggregator.forward(state, time=0.0, dt=1800.0)
        assert "air_temperature_hillslope" in updates
        assert "moisture_hillslope" in updates
        assert updates["_hierarchical_diagnostics"]["n_variables"] == 2

    def test_warning_on_missing_variable(self, aggregator: HierarchicalAggregator) -> None:
        """Aggregator emits diagnostics when requested variable is absent."""
        data: Dict[str, torch.Tensor] = {"air_temperature": torch.ones(6)}
        result, audit = aggregator.aggregate(data=data, variables=["air_temperature", "missing_var"])

        assert "air_temperature" in result
        assert "missing_var" not in result
        assert any("missing_var" in warning for warning in audit["warnings"])

    def test_aggregation_failure_raises(self, aggregator: HierarchicalAggregator, monkeypatch: pytest.MonkeyPatch) -> None:
        """Aggregation raises when strategy computation fails."""
        data = {"air_temperature": torch.ones(6)}

        def _boom(*_args, **_kwargs):
            raise RuntimeError("forced failure for test")

        monkeypatch.setitem(aggregator.method_map, "area_weighted_mean", _boom)

        with pytest.raises(RuntimeError, match="Aggregation failed"):
            aggregator.aggregate(data=data, variables=["air_temperature"], strategy="area_weighted_mean")


def test_weight_type_and_rule_are_required_method_choices(hierarchy_index: HierarchyIndex) -> None:
    """MA-21 item 204: the weighting rule and the scaling rule select the aggregation method, so
    HierarchicalAggregator has no default for either (nor an alternate spelling of them); every
    caller states them."""
    import inspect

    parameters = inspect.signature(HierarchicalAggregator.__init__).parameters
    for name in ("weight_type", "rule"):
        assert parameters[name].default is inspect.Parameter.empty, name
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    for alias in ("aggregation_strategy", "scaling_mode"):
        assert alias not in parameters, alias
    with pytest.raises(TypeError, match="weight_type"):
        HierarchicalAggregator(index=hierarchy_index, rule="conservative")
    with pytest.raises(TypeError, match="rule"):
        HierarchicalAggregator(index=hierarchy_index, weight_type="area")
    aggregator = HierarchicalAggregator(index=hierarchy_index, weight_type="area", rule="conservative")
    assert (aggregator.weight_type, aggregator.rule) == ("area", "conservative")


def test_unknown_keyword_arguments_are_rejected(hierarchy_index: HierarchyIndex) -> None:
    """MA-21 item 209: the constructor accepts its declared parameters only; an unknown keyword
    (a misspelled option or an option of another class) raises TypeError instead of being
    swallowed."""
    import inspect

    kinds = {parameter.kind for parameter in inspect.signature(HierarchicalAggregator.__init__).parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds
    with pytest.raises(TypeError, match="unexpected_option"):
        HierarchicalAggregator(
            index=hierarchy_index, weight_type="area", rule="conservative", unexpected_option=1
        )

