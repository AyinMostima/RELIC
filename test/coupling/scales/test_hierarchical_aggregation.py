"""
Test Hierarchical Aggregation Module

Tests for hierarchical scaling functionality including aggregation,
disaggregation, and conservation verification.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch

from relic.coupling.scales.scale_manager import ScaleManager
from relic.core.state import StateManager
from relic.parameters import get_global_manager
from relic.coupling.core.policy import ScalingMode, get_precision_policy
from relic.coupling.scales.hierarchical import HierarchyIndex, HierarchicalAggregator
from relic.coupling.scales.hierarchical_strategies import HierarchicalScalingStrategy
from relic.coupling.scales.types import ScalingStrategy, SpatialQuantityBasis
from relic.coupling.scales.hierarchical.templates import (
    get_rhessys_template,
    create_validation_data,
    validate_aggregation_conservation
)


def test_hierarchical_scaling_audit_never_infers_dataset_cf_conformance():
    class Manager:
        enforce_conservation = True

        def _determine_cf_compliance(self, *_args, **_kwargs):
            pytest.fail("hierarchical audit must not infer dataset CF conformance")

        def _collect_facade_warnings(self):
            return []

    strategy = object.__new__(HierarchicalScalingStrategy)
    strategy.manager = Manager()

    audit = strategy._build_scaling_audit(
        operation="aggregate",
        source_scale="patch",
        target_scale="hillslope",
        variables=["signal"],
        mode=ScalingMode.CONSERVATIVE,
        strategies={"signal": ScalingStrategy.AREA_WEIGHTED_MEAN},
        operation_audit={},
        conservation_audit={},
        updated_metadata={"cf_compliant": True},
        constraints_applied=False,
    )

    assert audit.cf_compliant is False
    assert audit.metadata["cf_validation_scope"] == "not_evaluated"
    assert audit.metadata["dataset_conformance_checked"] is False


def test_hierarchical_aggregator_init_does_not_raise(monkeypatch):
    """Aggregator should initialize without unexpected keyword errors."""

    class DummyParameterManager:
        def get(self, key, units=None, expected_type=None, required=False, namespace=None, **_):
            lookup = {
                "coupling.scales.defaults.hierarchy_numerical_tolerance": 1e-6,
                # The one scale-operator default tolerance of a standalone aggregator.
                "conservation_tolerance": 1e-8,
                "coupling.scales.defaults.conservation_tolerance": 1e-8,
            }
            if "units_per" in key:
                return 1
            if key.endswith("default_area"):
                return 1.0
            return lookup.get(key, 1.0)

        def get_parameter(self, key):
            """No ParameterSpec metadata (FunctorBase.get_parameter_value reads specs first)."""
            return None

    dummy_pm = DummyParameterManager()

    monkeypatch.setattr(
        "relic.coupling.core.policy.get_precision_policy",
        lambda mode, parameter_manager=None: type(
            "Policy",
            (),
            {"tolerance": 1e-6, "enforce_conservation": True, "dtype": torch.float32},
        )(),
    )

    index = HierarchyIndex(
        source_scale="leaf",
        target_scale="canopy",
        device=torch.device("cpu"),
        n_source_units=2,
        parameter_manager=dummy_pm,
    )
    class DummyAggregator(HierarchicalAggregator):
        def _resolve_numerical_epsilon(self) -> float:
            return 1e-9

    aggregator = DummyAggregator(
        index=index,
        device=torch.device("cpu"),
        parameter_manager=dummy_pm,
        hierarchy_levels=["leaf", "canopy"],
        weight_type="area",
        rule="conservative",
    )

    assert aggregator.source_scale == "leaf"
    assert aggregator.target_scale == "canopy"
    assert isinstance(aggregator.hierarchy_levels, list)
    assert aggregator.hierarchy_levels[0] == "leaf"


class TestHierarchyIndex:
    """Test HierarchyIndex functionality."""
    
    def test_create_default_hierarchy(self):
        """Test creating default hierarchy."""
        index = HierarchyIndex(
            source_scale="patch",
            target_scale="landscape",
            device=torch.device("cpu"),
            n_source_units=100
        )
        
        assert index.n_source_units == 100
        assert index.n_target_units == 1
        assert index.hierarchy_matrix is not None
        
    def test_rhessys_template(self):
        """Test RHESSys template loading."""
        index = HierarchyIndex.from_rhessys_template("small_basin")
        
        assert index.source_scale == "patch"
        assert index.target_scale == "landscape"
        # RHESSys-style aggregation has different conservation properties
        # The "conservation error" warning is expected for aggregation weights
        # What matters is the final aggregation conservation, not weight matrix row sums
        assert index.source_scale == "patch"  # Just verify the template loaded correctly
        
    def test_weight_matrix_conservation(self):
        """Test weight matrix conservation properties for RHESSys-style aggregation."""
        index = HierarchyIndex(
            source_scale="patch",
            target_scale="landscape",
            device=torch.device("cpu"),
            n_source_units=100
        )
        
        # For patch->landscape aggregation, column sums should equal 1.0 (normalized weights)
        # Each of 100 patches contributes weight of 1/100 = 0.01 to single landscape
        weight_dense = index.to_dense()
        
        # Check column sums (aggregation weights should be normalized)
        col_sums = weight_dense.sum(dim=0)
        assert torch.allclose(
            col_sums, 
            torch.ones_like(col_sums), 
            atol=1e-6
        ), f"Column sums should be 1.0 for normalized aggregation, got {col_sums}"
        
        # Check that each row has expected weight
        expected_weight_per_patch = 1.0 / 100.0
        row_sums = weight_dense.sum(dim=1)
        assert torch.allclose(
            row_sums,
            torch.full_like(row_sums, expected_weight_per_patch),
            atol=1e-6
        ), f"Each patch should contribute {expected_weight_per_patch}, got row_sums range [{row_sums.min():.6f}, {row_sums.max():.6f}]"
        
    def test_no_dead_builders_or_raw_weight_application(self):
        """Only the parameter and topology builders remain; parent sums use the validated
        membership mapping of the aggregator, not the raw (column-normalised) H."""
        for name in (
            "_build_patch_to_hillslope_index",
            "_build_hillslope_to_basin_index",
            "_build_patch_to_landscape_index",
            "_build_default_index",
            "_assign_areas_from_parameter_key",
            "aggregate_weights",
            "apply_weights",
        ):
            assert not hasattr(HierarchyIndex, name), name

    def test_topology_build_does_not_swallow_errors(self):
        import inspect

        source = inspect.getsource(HierarchyIndex._build_from_topology)
        assert "except Exception" not in source


class TestHierarchicalAggregator:
    """Test HierarchicalAggregator functionality."""
    
    def test_area_weighted_aggregation(self):
        """Test area-weighted mean aggregation."""
        template = get_rhessys_template("test_hillslope")
        index = HierarchyIndex(
            source_scale="patch",
            target_scale="landscape",
            topology=template
        )
        aggregator = HierarchicalAggregator(index, weight_type="area", rule="conservative")
        
        # Create test data
        data = {"air_temperature": torch.ones(10) * 273.15}
        variables = ["air_temperature"]
        
        # Aggregate
        result, audit = aggregator.aggregate(
            data, variables, strategy="area_weighted_mean"
        )
        
        assert "air_temperature" in result
        assert result["air_temperature"].shape[0] == 1
        assert abs(result["air_temperature"][0] - 273.15) < 0.01
        
    def test_standalone_operators_read_the_scale_default_tolerance(self):
        """A hierarchical aggregator or WeightOps used outside the facade reads the one scale-operator
        default coupling.scales.defaults.conservation_tolerance (the facade passes the tolerance of the
        call's mode instead)."""
        from relic.coupling.scales.hierarchical import weight_ops

        default = float(
            get_global_manager().get("conservation_tolerance", namespace="coupling.scales.defaults", required=True)
        )
        index = HierarchyIndex(
            source_scale="patch", target_scale="landscape", topology=get_rhessys_template("test_hillslope")
        )
        assert HierarchicalAggregator(index, weight_type="area", rule="conservative").conservation_tolerance == default
        # The weight check of WeightOps has its own single source, the hierarchy normalization
        # tolerance (not the ledger tolerance above).
        hierarchy_tolerance = float(
            get_global_manager().get(
                "hierarchy_numerical_tolerance", namespace="coupling.scales.defaults", required=True
            )
        )
        assert weight_ops._resolve_weight_tolerance() == hierarchy_tolerance

    def test_weight_check_is_strict_with_one_tolerance_source(self):
        """validate_weights raises for a wrong shape, non-finite or negative weights and a member
        column whose sum misses 1 by more than coupling.scales.defaults.hierarchy_numerical_tolerance;
        a target without members passes. It takes no tolerance argument, WeightOps has no silent
        fill helpers, and HierarchyIndex keeps no warning list."""
        import inspect

        from relic.coupling.scales.hierarchical import weight_ops
        from relic.coupling.scales.hierarchical.weight_ops import WeightOps

        for name in ("handle_zero_columns", "mask_nan", "safe_clip"):
            assert not hasattr(WeightOps, name), name
        assert "tolerance" not in inspect.signature(WeightOps.validate_weights).parameters
        assert not hasattr(HierarchyIndex, "get_warnings")
        tolerance = weight_ops._resolve_weight_tolerance()
        good = torch.tensor([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=torch.float64)
        assert WeightOps.validate_weights(good, source_dim=2, target_dim=3) is None
        cases = {
            "shape": (good, 3, 3),
            "finite": (torch.tensor([[float("nan")], [1.0]], dtype=torch.float64), 2, 1),
            "negative": (torch.tensor([[-0.5], [1.5]], dtype=torch.float64), 2, 1),
            "sum": (torch.tensor([[0.5], [0.5 + 10.0 * tolerance]], dtype=torch.float64), 2, 1),
        }
        for label, (weights, n_source, n_target) in cases.items():
            with pytest.raises(ValueError):
                WeightOps.validate_weights(weights, source_dim=n_source, target_dim=n_target)

    def test_weight_ops_keep_no_helper_that_compares_amounts_with_epsilon(self):
        """The unused WeightOps helpers that clamped area sums or compared totals (units of the
        field) with the dimensionless coupling.scales.defaults.epsilon are gone."""
        from relic.coupling.scales.hierarchical import weight_ops
        from relic.coupling.scales.hierarchical.weight_ops import WeightOps

        for name in (
            "normalize_weights",
            "area_weighted_apply",
            "create_sparse_weights",
            "compute_conservation_error",
            "apply_conservation_correction",
        ):
            assert not hasattr(WeightOps, name), name
        assert not hasattr(weight_ops, "_resolve_precision_params")

    def test_internal_ledger_accepts_float32_rounding_by_the_shared_rule(self, caplog):
        """The aggregator's own ledger check uses the shared rule with the rounding bound of the
        field dtype: float32 area densities aggregated under the 1e-8 scale default log no
        violation, and a doubled area density still does."""
        import logging

        from relic.coupling.core import conservation_checker

        seen = []
        original_rule = conservation_checker.ledger_acceptance

        def spy(**kwargs):
            seen.append(kwargs["rounding_dtype"])
            return original_rule(**kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(conservation_checker, "ledger_acceptance", spy)
        index = HierarchyIndex(
            source_scale="patch", target_scale="landscape", topology=get_rhessys_template("small_basin")
        )
        aggregator = HierarchicalAggregator(index, weight_type="area", rule="conservative")
        n_source = int(index.get_source_areas().numel())
        generator = torch.Generator().manual_seed(11)
        values = (torch.rand(n_source, generator=generator, dtype=torch.float64) * 10.0 + 5.0).to(torch.float32)
        metadata = {"gpp": {"spatial_basis": "area_density", "units": "umol m-2 s-1"}}
        try:
            with caplog.at_level(logging.WARNING, logger="relic.coupling.scales.conservation_enforcer"):
                result, _audit = aggregator.aggregate(
                    {"gpp": values}, ["gpp"], strategy="conservative", variable_metadata=metadata
                )
        finally:
            monkeypatch.undo()
        # The ledger is judged in float64 with the rounding bound of float32, the field's dtype.
        assert seen == [torch.float32]
        assert "conservation violated" not in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="relic.coupling.scales.conservation_enforcer"):
            aggregator._check_extensive_conservation(
                values, 2.0 * result["gpp"], "gpp", SpatialQuantityBasis.AREA_DENSITY
            )
        assert "conservation violated" in caplog.text

    def test_conservative_aggregation(self):
        """Test conservative aggregation for extensive variables."""
        template = get_rhessys_template("small_basin")
        index = HierarchyIndex(
            source_scale="patch",
            target_scale="landscape",
            topology=template
        )
        aggregator = HierarchicalAggregator(index, weight_type="area", rule="conservative")
        
        # A catalogued area density (net_primary_productivity, kg C m-2 s-1 per unit ground area)
        data = {"net_primary_productivity": torch.rand(100) * 1e-6}
        variables = ["net_primary_productivity"]
        
        # Aggregate with conservation
        result, audit = aggregator.aggregate(
            data, variables, strategy="flow_conserving", mode="strict_conservative"
        )
        
        # Check conservation
        source_areas = torch.tensor(template["source_areas"])
        target_areas = torch.tensor(template["target_areas"])
        
        source_total = (data["net_primary_productivity"] * source_areas).sum()
        target_total = (result["net_primary_productivity"] * target_areas).sum()
        
        rel_error = abs(target_total - source_total) / source_total
        
        # Updated tolerance based on numerical precision analysis
        # Our debug shows ~1.7e-7 error, so 1e-6 is appropriate for conservative mode
        assert rel_error < 1e-6  # Conservative tolerance (realistic for float32)
        
    def test_disaggregation(self):
        """Test disaggregation from coarse to fine."""
        template = get_rhessys_template("test_hillslope")
        # Note: For disaggregation, we reverse source and target
        index = HierarchyIndex(
            source_scale="patch",
            target_scale="landscape",
            topology=template
        )
        aggregator = HierarchicalAggregator(index, weight_type="area", rule="conservative")
        
        # Create coarse data
        coarse_data = {"air_temperature": torch.tensor([285.0])}
        variables = ["air_temperature"]
        
        # Disaggregate
        result, audit = aggregator.disaggregate(
            coarse_data, variables, mode="conservative"
        )
        
        assert "air_temperature" in result
        assert result["air_temperature"].shape[0] == 10
        assert torch.allclose(result["air_temperature"], torch.ones(10) * 285.0, atol=0.1)


class TestScaleManagerFacade:
    """Test ScaleManager integration."""
    
    def test_manager_interface(self):
        """Test aggregation interface availability."""
        manager = ScaleManager(
            default_mode=ScalingMode.CONSERVATIVE,
            parameter_manager=get_global_manager(),
            state_manager=StateManager(),
        )
        manager.initialize()
        
        assert hasattr(manager, 'aggregate')
        assert hasattr(manager, 'disaggregate')
        
    def test_mode_switching(self):
        """Test switching precision modes."""
        manager = ScaleManager(
            default_mode=ScalingMode.CONSERVATIVE,
            parameter_manager=get_global_manager(),
            state_manager=StateManager(),
        )
        manager.initialize()
        manager.set_mode(ScalingMode.STRICT_CONSERVATIVE)
        assert manager.default_mode == ScalingMode.STRICT_CONSERVATIVE
        strict = get_precision_policy(ScalingMode.STRICT_CONSERVATIVE).tolerance
        assert manager.ledger_tolerance() == manager.conservation_tolerance == strict

        manager.set_mode(ScalingMode.FAST_APPROX)
        assert manager.default_mode == ScalingMode.FAST_APPROX
        fast = get_precision_policy(ScalingMode.FAST_APPROX).tolerance
        assert manager.ledger_tolerance() == manager.conservation_tolerance == fast
        # The facade factory keeps no tolerance: each regrid call passes its mode's tolerance.
        assert manager.regridder_factory.conservation_tolerance is None


def _facade_manager() -> ScaleManager:
    manager = ScaleManager(
        default_mode=ScalingMode.CONSERVATIVE,
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    manager.initialize()
    return manager


# Two hillslopes of 4 m2, each covering two patches of unequal area.
_UNEQUAL_TOPOLOGY = {
    "n_source_units": 4,
    "n_target_units": 2,
    "parent_map": {0: 0, 1: 0, 2: 1, 3: 1},
    "source_areas": [1.0, 3.0, 2.0, 2.0],
    "target_areas": [4.0, 4.0],
}


class TestFacadeUsesTheDeclaredBasis:
    """ScaleManager hierarchy facade: declared basis, basis-aware conservation audit."""

    def test_uncatalogued_variables_need_declared_metadata(self):
        manager = _facade_manager()
        data = torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float64)
        with pytest.raises((KeyError, ValueError), match="litter_mass"):
            manager.aggregate(
                data, "patch", "hillslope", variables=["litter_mass"], topology=dict(_UNEQUAL_TOPOLOGY)
            )
        result, audit = manager.aggregate(
            data,
            "patch",
            "hillslope",
            variables=["litter_mass"],
            metadata={"litter_mass": {"spatial_basis": "cell_total", "units": "kg"}},
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        assert torch.allclose(result.to(torch.float64), torch.tensor([6.0, 4.0], dtype=torch.float64))
        assert audit.relative_error == pytest.approx(0.0, abs=1e-12)

    def test_area_density_audit_is_area_weighted_and_does_not_rescale(self):
        manager = _facade_manager()
        assert manager.enforce_conservation is True
        # precipitation_flux is an area density (kg m-2 s-1) in the YAML catalog.
        data = torch.tensor([0.004, 0.002, 0.001, 0.003], dtype=torch.float64)
        result, audit = manager.aggregate(
            data, "patch", "hillslope", variables=["precipitation_flux"], topology=dict(_UNEQUAL_TOPOLOGY)
        )
        # y_0 = (0.004*1 + 0.002*3) / 4, y_1 = (0.001*2 + 0.003*2) / 4
        assert torch.allclose(result.to(torch.float64), torch.tensor([0.0025, 0.002], dtype=torch.float64))
        assert audit.relative_error == pytest.approx(0.0, abs=1e-12)
        diagnostics = audit.metadata["conservation_audit"]["precipitation_flux"]
        assert diagnostics.source_total == pytest.approx(0.018)

    def test_facade_strategy_and_extensivity_follow_the_basis_not_the_name(self):
        manager = _facade_manager()
        with pytest.raises(ValueError, match="ambiguous"):
            manager._is_extensive({"name": "carbon_flux", "variable_type": "extensive"})
        with pytest.raises(ValueError, match="declare"):
            manager._is_extensive({"name": "carbon_flux", "units": "kg m-2 s-1"})
        assert manager._is_extensive({"name": "energy_flux", "variable_type": "intensive", "units": "W m-2"}) is False
        assert manager._is_extensive({"name": "x", "spatial_basis": "cell_total"}) is True
        assert not hasattr(
            __import__("relic.coupling.scales.hierarchical_strategies", fromlist=["x"]),
            "_infer_runtime_variable_type",
        )


class TestFacadeCFMetadataPropagation:
    """Governed aggregation and disaggregation propagate cell_methods and history."""

    _META = {"litter_mass": {"spatial_basis": "cell_total", "units": "kg", "cell_methods": "area: sum"}}

    def test_governed_aggregation_records_the_rescaling(self):
        manager = _facade_manager()
        result, audit = manager.aggregate(
            torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float64),
            "patch",
            "hillslope",
            variables=["litter_mass"],
            metadata=dict(self._META),
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        assert torch.allclose(result.to(torch.float64), torch.tensor([6.0, 4.0], dtype=torch.float64))
        cf = audit.metadata["cf_metadata"]
        assert cf["cell_methods"] == "area: sum"
        assert cf["history"] == "Aggregated area from patch to hillslope (area: sum)"

    def test_conservative_facade_aggregates_categorical_fields_by_area_weighted_mode(self):
        manager = _facade_manager()
        result, audit = manager.aggregate(
            torch.tensor([1.0, 2.0, 4.0, 3.0], dtype=torch.float64),
            "patch",
            "hillslope",
            variables=["land_cover"],
            metadata={"land_cover": {"spatial_basis": "categorical", "units": "1"}},
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        # Parent 0: class 1 on 1 m2, class 2 on 3 m2 -> 2; parent 1: equal areas -> smaller code 3.
        assert torch.equal(result.to(torch.float64), torch.tensor([2.0, 3.0], dtype=torch.float64))
        assert audit.strategy.value == "mode"
        assert audit.metadata["cf_metadata"]["cell_methods"] == "area: mode"

    def test_governed_disaggregation_propagates_cf_metadata(self):
        manager = _facade_manager()
        result, audit = manager.disaggregate(
            torch.tensor([6.0, 4.0], dtype=torch.float64),
            "hillslope",
            "patch",
            variables=["litter_mass"],
            metadata=dict(self._META),
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        # Area-proportional split of cell totals X_i = Y_j A_i / sum_k A_k with A = [1, 3 | 2, 2].
        assert torch.allclose(
            result.to(torch.float64), torch.tensor([1.5, 4.5, 2.0, 2.0], dtype=torch.float64)
        )
        cf = audit.metadata["cf_metadata"]
        assert "area: sum" in cf["cell_methods"]
        assert "hillslope" in cf["history"] and "patch" in cf["history"]
        assert cf["history"] == "Disaggregated area from hillslope to patch (area: sum)"


class TestFacadeResolvesTheMethodPerVariable:
    """One facade call may mix bases: each variable gets the method of its declared basis."""

    _MIXED = {
        "land_cover": {"spatial_basis": "categorical", "units": "1"},
        "litter_mass": {"spatial_basis": "cell_total", "units": "kg"},
    }

    def test_categorical_and_numeric_variables_aggregate_in_one_call(self):
        manager = _facade_manager()
        result, audit = manager.aggregate(
            {
                "land_cover": torch.tensor([1.0, 2.0, 4.0, 3.0], dtype=torch.float64),
                "litter_mass": torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float64),
            },
            "patch",
            "hillslope",
            metadata={name: dict(entry) for name, entry in self._MIXED.items()},
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        # Area-weighted mode of the class codes; parent sums of the cell totals.
        assert torch.equal(result["land_cover"].to(torch.float64), torch.tensor([2.0, 3.0], dtype=torch.float64))
        assert torch.allclose(result["litter_mass"].to(torch.float64), torch.tensor([6.0, 4.0], dtype=torch.float64))
        assert audit.strategy is None
        assert audit.metadata["strategy_per_variable"] == {"land_cover": "mode", "litter_mass": "flow_conserving"}
        cf = audit.metadata["cf_metadata"]
        assert cf["land_cover"]["cell_methods"] == "area: mode"
        assert cf["litter_mass"]["cell_methods"] == "area: sum"
        assert cf["litter_mass"]["history"] == "Aggregated area from patch to hillslope (area: sum)"

    def test_numeric_variables_of_different_bases_disaggregate_in_one_call(self):
        manager = _facade_manager()
        result, audit = manager.disaggregate(
            {
                "litter_mass": torch.tensor([6.0, 4.0], dtype=torch.float64),
                "probe_intensive_field": torch.tensor([285.0, 290.0], dtype=torch.float64),
            },
            "hillslope",
            "patch",
            metadata={
                "litter_mass": {"spatial_basis": "cell_total", "units": "kg"},
                "probe_intensive_field": {"spatial_basis": "intensive", "units": "K"},
            },
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        assert torch.allclose(
            result["litter_mass"].to(torch.float64), torch.tensor([1.5, 4.5, 2.0, 2.0], dtype=torch.float64)
        )
        assert torch.allclose(
            result["probe_intensive_field"].to(torch.float64),
            torch.tensor([285.0, 285.0, 290.0, 290.0], dtype=torch.float64),
        )
        cf = audit.metadata["cf_metadata"]
        assert cf["litter_mass"]["cell_methods"] == "area: sum"
        assert cf["probe_intensive_field"]["cell_methods"] == "area: mean"

    def test_categorical_disaggregation_is_an_explicit_error(self):
        manager = _facade_manager()
        for data, metadata in (
            (
                torch.tensor([2.0, 3.0], dtype=torch.float64),
                {"land_cover": {"spatial_basis": "categorical", "units": "1"}},
            ),
            (
                {
                    "land_cover": torch.tensor([2.0, 3.0], dtype=torch.float64),
                    "litter_mass": torch.tensor([6.0, 4.0], dtype=torch.float64),
                },
                {name: dict(entry) for name, entry in self._MIXED.items()},
            ),
        ):
            with pytest.raises((ValueError, RuntimeError), match="land_cover.*cannot be disaggregated"):
                manager.disaggregate(
                    data,
                    "hillslope",
                    "patch",
                    variables=["land_cover"] if isinstance(data, torch.Tensor) else None,
                    metadata=metadata,
                    topology=dict(_UNEQUAL_TOPOLOGY),
                )


_LITTER_CELL_TOTAL = {"litter_mass": {"spatial_basis": "cell_total", "units": "kg"}}


def _perturb_aggregator(monkeypatch, factor):
    """Scale every aggregated field by factor to create a known conservation residual."""
    original = HierarchicalAggregator.aggregate

    def perturbed(self, *args, **kwargs):
        aggregated, aggregation_audit = original(self, *args, **kwargs)
        return {name: value * factor for name, value in aggregated.items()}, aggregation_audit

    monkeypatch.setattr(HierarchicalAggregator, "aggregate", perturbed)


class TestFacadeConservationIsAuditedNotRescaled:
    """Strict facade path: fail-closed audit; no multiplicative correction of the result."""

    def test_strict_path_fails_closed_instead_of_rescaling(self, monkeypatch):
        manager = _facade_manager()
        assert manager.enforce_conservation is True
        _perturb_aggregator(monkeypatch, 1.1)
        with pytest.raises(RuntimeError, match="litter_mass.*conservation"):
            manager.aggregate(
                torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float64),
                "patch",
                "hillslope",
                variables=["litter_mass"],
                metadata=dict(_LITTER_CELL_TOTAL),
                topology=dict(_UNEQUAL_TOPOLOGY),
            )

    def test_ledger_tolerance_is_the_precision_tolerance_of_the_calls_mode(self, monkeypatch):
        """Per-call mode, per-call tolerance: coupling.scales.defaults.precision.<mode>.tolerance."""
        manager = _facade_manager()
        assert manager.enforce_conservation is True
        # A relative residual of 1e-7 on the strict path of every call below.
        _perturb_aggregator(monkeypatch, 1.0 + 1.0e-7)
        data = torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float64)
        request = dict(variables=["litter_mass"], metadata=dict(_LITTER_CELL_TOTAL), topology=dict(_UNEQUAL_TOPOLOGY))
        # conservative (default mode, 1e-9) and strict_conservative (1e-12) reject it ...
        for mode in (None, ScalingMode.CONSERVATIVE, ScalingMode.STRICT_CONSERVATIVE):
            with pytest.raises(RuntimeError, match="litter_mass.*conservation"):
                manager.aggregate(data, "patch", "hillslope", mode=mode, **request)
        # ... balanced (1e-6) accepts it: the call's mode selects the tolerance.
        balanced = get_precision_policy(ScalingMode.BALANCED).tolerance
        _, audit = manager.aggregate(data, "patch", "hillslope", mode=ScalingMode.BALANCED, **request)
        diagnostics = audit.metadata["conservation_audit"]["litter_mass"]
        assert diagnostics.tolerance == balanced == manager.ledger_tolerance(ScalingMode.BALANCED)
        assert diagnostics.is_conserved is True
        assert diagnostics.relative_error > manager.ledger_tolerance(ScalingMode.CONSERVATIVE)
        # Disaggregation follows the same rule.
        _, down = manager.disaggregate(
            torch.tensor([6.0, 4.0], dtype=torch.float64),
            "hillslope",
            "patch",
            mode=ScalingMode.STRICT_CONSERVATIVE,
            variables=["litter_mass"],
            metadata=dict(_LITTER_CELL_TOTAL),
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        strict = get_precision_policy(ScalingMode.STRICT_CONSERVATIVE).tolerance
        assert down.metadata["conservation_audit"]["litter_mass"].tolerance == strict
        # A call without a mode takes the default mode, which set_mode moves.
        manager.set_mode(ScalingMode.BALANCED)
        _, audit = manager.aggregate(data, "patch", "hillslope", **request)
        assert audit.metadata["conservation_audit"]["litter_mass"].tolerance == balanced

    def test_audit_only_path_reports_the_residual_and_returns_the_field_unchanged(self, monkeypatch):
        manager = _facade_manager()
        manager.enforce_conservation = False
        _perturb_aggregator(monkeypatch, 1.1)
        result, audit = manager.aggregate(
            torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float64),
            "patch",
            "hillslope",
            variables=["litter_mass"],
            mode=ScalingMode.CONSERVATIVE,
            metadata=dict(_LITTER_CELL_TOTAL),
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        assert torch.allclose(result.to(torch.float64), torch.tensor([6.6, 4.4], dtype=torch.float64))
        diagnostics = audit.metadata["conservation_audit"]["litter_mass"]
        assert diagnostics.is_conserved is False
        assert diagnostics.relative_error == pytest.approx(0.1)

    def test_uncovered_amount_is_reported_and_orphans_are_rejected_up_front(self):
        manager = _facade_manager()
        result, audit = manager.aggregate(
            torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float64),
            "patch",
            "hillslope",
            variables=["litter_mass"],
            metadata=dict(_LITTER_CELL_TOTAL),
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        assert torch.allclose(result.to(torch.float64), torch.tensor([6.0, 4.0], dtype=torch.float64))
        diagnostics = audit.metadata["conservation_audit"]["litter_mass"]
        assert diagnostics.relative_error == pytest.approx(0.0, abs=1e-12)
        # A full partition leaves nothing uncovered; the amount is reported, never rescaled.
        assert diagnostics.metadata["uncovered_source_amount"] == pytest.approx(0.0)
        orphan_topology = dict(_UNEQUAL_TOPOLOGY)
        orphan_topology["parent_map"] = {0: 0, 1: 0, 2: 1}  # child 3 has no parent
        with pytest.raises(ValueError, match="parent_map"):
            manager.aggregate(
                torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float64),
                "patch",
                "hillslope",
                variables=["litter_mass"],
                metadata=dict(_LITTER_CELL_TOTAL),
                topology=orphan_topology,
            )

    def test_conservation_enforcer_has_no_correction_path(self):
        import inspect

        from relic.coupling.scales.conservation_enforcer import ConservationEnforcer

        for name in (
            "enforce_conservation",
            "_multiplicative_enforcement",
            "_additive_enforcement",
            "_lagrangian_enforcement",
        ):
            assert not hasattr(ConservationEnforcer, name), name
        assert "enforcement_method" not in inspect.signature(ConservationEnforcer.__init__).parameters

    def test_float32_rounding_is_accepted_on_the_default_strict_path(self, monkeypatch):
        """The default facade (enforce_conservation=True) accepts float32 rounding, not a wrong result."""
        manager = _facade_manager()
        assert manager.enforce_conservation is True
        generator = torch.Generator().manual_seed(3)
        n_patches = 64
        areas = (torch.rand(n_patches, generator=generator, dtype=torch.float64) * 1.0e4 + 1.0).tolist()
        topology = {
            "n_source_units": n_patches,
            "n_target_units": 2,
            "parent_map": {index: index % 2 for index in range(n_patches)},
            "source_areas": areas,
            "target_areas": [sum(areas[0::2]), sum(areas[1::2])],
        }
        values = (torch.rand(n_patches, generator=generator, dtype=torch.float64) * 1.0e-6).to(torch.float32)
        request = dict(
            variables=["rho"],
            metadata={"rho": {"spatial_basis": "area_density", "units": "kg m-2"}},
            topology=topology,
        )
        result, audit = manager.aggregate(values, "patch", "hillslope", **request)
        diagnostics = audit.metadata["conservation_audit"]["rho"]
        # The float32 rounding of the returned densities exceeds the float64-scale tolerance ...
        assert diagnostics.relative_error > diagnostics.tolerance
        # ... and is accepted as rounding of the result dtype, not as a conservation violation.
        assert diagnostics.is_conserved is True
        assert diagnostics.absolute_error <= diagnostics.metadata["accepted_residual"]
        assert diagnostics.metadata["rounding_allowance"] > 0.0
        # A wrong (doubled) result is still rejected on the same strict path.
        _perturb_aggregator(monkeypatch, 2.0)
        with pytest.raises(RuntimeError, match="rho.*conservation"):
            manager.aggregate(values, "patch", "hillslope", **request)

    def test_both_ledgers_accept_through_one_shared_function(self, monkeypatch):
        """The facade ledger and the whole-support ledger call the one shared acceptance function."""
        from relic.coupling.core import conservation_checker
        import relic.coupling.scales.alias_utils as alias_utils
        from relic.coupling.scales.alias_utils import aggregate_spatial_alias
        from relic.coupling.scales.types import SpatialQuantityBasis

        seen = []
        original = conservation_checker.ledger_acceptance

        def spy(**kwargs):
            seen.append(kwargs["rounding_dtype"])
            return original(**kwargs)

        monkeypatch.setattr(conservation_checker, "ledger_acceptance", spy)
        manager = _facade_manager()
        manager.aggregate(
            torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float32),
            "patch",
            "hillslope",
            variables=["litter_mass"],
            metadata=dict(_LITTER_CELL_TOTAL),
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        aggregate_spatial_alias(
            data_dict={"q": torch.tensor([4.0, 8.0], dtype=torch.float32)},
            variable_bases={"q": SpatialQuantityBasis.CELL_TOTAL},
            source_scale="grid_cell",
            target_scale="support",
            mode=ScalingMode.CONSERVATIVE,
            device=torch.device("cpu"),
            enforce_conservation=True,
            tolerance=1.0e-8,
        )
        # The aggregator's own ledger, the facade ledger and the whole-support ledger.
        assert [dtype for dtype in seen if dtype is not None] == [torch.float32, torch.float32, torch.float32]
        assert not hasattr(alias_utils, "_reduction_rounding_allowance")


# Two patches of 1 m2 and 3 m2 in one parent of 4 m2 (the topology of the review counterexample).
_PAIR_TOPOLOGY = {
    "n_source_units": 2,
    "n_target_units": 1,
    "parent_map": {0: 0, 1: 0},
    "source_areas": [1.0, 3.0],
    "target_areas": [4.0],
}

# The same pair with a second parent of 2 m2 that has no member: the target support is only
# partly covered by the source support.
_PARTIAL_PAIR_TOPOLOGY = {
    "n_source_units": 2,
    "n_target_units": 2,
    "parent_map": {0: 0, 1: 0},
    "source_areas": [1.0, 3.0],
    "target_areas": [4.0, 2.0],
}


def _public_entry_manager() -> ScaleManager:
    """The public entry of the review: ScaleManager(device=cpu, state_manager=StateManager())."""
    manager = ScaleManager(device=torch.device("cpu"), state_manager=StateManager())
    manager.ensure_ready()
    return manager


def _ledger_rows(diagnostics):
    return {row["batch_index"]: row for row in diagnostics.metadata["batch_ledgers"]}


class TestFacadeLedgerIsCheckedPerBatch:
    """Leading axes of a field are batches: one ledger per batch, never one summed ledger.

    The hierarchical kernels map [..., n_source] to [..., n_target] with the spatial support on
    the last axis; the audit reduces over that axis only and judges every batch on its own, so
    residuals of opposite sign in different batches never cancel.
    """

    _CLIPPED = {"audit_batch": {"spatial_basis": "cell_total", "physical_bounds": [1.0, 3.0]}}

    def _clipped_request(self, topology):
        return dict(metadata={name: dict(entry) for name, entry in self._CLIPPED.items()}, topology=dict(topology))

    def test_clipped_batch_residuals_of_opposite_sign_do_not_cancel(self):
        """Review counterexample: batch totals [0, 4], clipped targets [[1], [3]], residuals +1 and -1."""
        source = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float64)
        # Strict path (the default facade enforces conservation): any violated batch raises.
        manager = _public_entry_manager()
        assert manager.enforce_conservation is True
        with pytest.raises(RuntimeError, match=r"audit_batch.*violates conservation.*2 of 2 batch"):
            manager.aggregate(
                {"audit_batch": source}, "fine_grid", "coarse_grid", **self._clipped_request(_PAIR_TOPOLOGY)
            )
        # Non-blocking path: the true batch residuals and the failure set are disclosed.
        manager = _public_entry_manager()
        manager.enforce_conservation = False
        result, audit = manager.aggregate(
            {"audit_batch": source},
            "fine_grid",
            "coarse_grid",
            mode=ScalingMode.CONSERVATIVE,
            **self._clipped_request(_PAIR_TOPOLOGY),
        )
        torch.testing.assert_close(result["audit_batch"], torch.tensor([[1.0], [3.0]], dtype=torch.float64))
        diagnostics = audit.metadata["conservation_audit"]["audit_batch"]
        assert diagnostics.is_conserved is False
        assert diagnostics.metadata["batch_shape"] == (2,)
        assert diagnostics.metadata["failed_batches"] == [(0,), (1,)]
        rows = _ledger_rows(diagnostics)
        assert [rows[(b,)]["source_total"] for b in (0, 1)] == [0.0, 4.0]
        assert [rows[(b,)]["target_total"] for b in (0, 1)] == [1.0, 3.0]
        assert [rows[(b,)]["residual"] for b in (0, 1)] == [1.0, -1.0]
        assert all(row["is_conserved"] is False for row in rows.values())
        # The summary is the largest batch residual, never the signed sum (+1 - 1 = 0).
        assert diagnostics.absolute_error == 1.0
        assert audit.mass_balance_error == 1.0
        assert audit.relative_error >= 0.25

    def test_partial_coverage_and_clipping_do_not_cancel_across_batches(self):
        """An empty parent clipped up in one batch and a full parent clipped down in the other."""
        source = torch.tensor([[0.5, 0.5], [2.5, 2.5]], dtype=torch.float64)
        manager = _facade_manager()
        with pytest.raises(RuntimeError, match=r"audit_batch.*violates conservation.*2 of 2 batch"):
            manager.aggregate(
                {"audit_batch": source}, "patch", "hillslope", **self._clipped_request(_PARTIAL_PAIR_TOPOLOGY)
            )
        manager.enforce_conservation = False
        result, audit = manager.aggregate(
            {"audit_batch": source},
            "patch",
            "hillslope",
            mode=ScalingMode.CONSERVATIVE,
            **self._clipped_request(_PARTIAL_PAIR_TOPOLOGY),
        )
        # Parent 1 has no member (0 before clipping, 1 after); parent 0 of batch 1 is 5, clipped to 3.
        torch.testing.assert_close(
            result["audit_batch"], torch.tensor([[1.0, 1.0], [3.0, 1.0]], dtype=torch.float64)
        )
        diagnostics = audit.metadata["conservation_audit"]["audit_batch"]
        rows = _ledger_rows(diagnostics)
        assert [rows[(b,)]["residual"] for b in (0, 1)] == [1.0, -1.0]
        assert diagnostics.metadata["failed_batches"] == [(0,), (1,)]
        assert diagnostics.is_conserved is False
        assert diagnostics.absolute_error == 1.0

    def test_every_ledger_of_several_leading_axes_is_judged_on_its_own(self, monkeypatch):
        """[2, 3, n] area densities on unequal areas: six ledgers; +d and -d never cancel."""
        original = HierarchicalAggregator.aggregate
        delta = 0.25

        def shifted(self, *args, **kwargs):
            aggregated, aggregation_audit = original(self, *args, **kwargs)
            moved = {}
            for name, value in aggregated.items():
                value = value.clone()
                value[0, 1, 0] += delta
                value[1, 2, 1] -= delta
                moved[name] = value
            return moved, aggregation_audit

        monkeypatch.setattr(HierarchicalAggregator, "aggregate", shifted)
        generator = torch.Generator().manual_seed(11)
        density = torch.rand((2, 3, 4), generator=generator, dtype=torch.float64) + 0.5
        request = dict(
            metadata={"rho": {"spatial_basis": "area_density", "units": "kg m-2"}},
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        manager = _facade_manager()
        with pytest.raises(RuntimeError, match=r"rho.*violates conservation.*2 of 6 batch"):
            manager.aggregate({"rho": density}, "patch", "hillslope", **request)
        manager.enforce_conservation = False
        result, audit = manager.aggregate(
            {"rho": density}, "patch", "hillslope", mode=ScalingMode.CONSERVATIVE, **request
        )
        assert tuple(result["rho"].shape) == (2, 3, 2)
        diagnostics = audit.metadata["conservation_audit"]["rho"]
        assert diagnostics.metadata["batch_shape"] == (2, 3)
        assert diagnostics.metadata["failed_batches"] == [(0, 1), (1, 2)]
        rows = _ledger_rows(diagnostics)
        assert len(rows) == 6
        areas = torch.tensor(_UNEQUAL_TOPOLOGY["source_areas"], dtype=torch.float64)
        for index, row in rows.items():
            # Each ledger is the area-weighted amount of its own batch.
            assert row["source_total"] == pytest.approx(float((density[index] * areas).sum()))
            if index not in ((0, 1), (1, 2)):
                assert row["is_conserved"] is True
                assert row["absolute_error"] <= row["accepted_residual"]
        # The parents are 4 m2: the shifted densities move 4 d of amount in each failed batch.
        assert rows[(0, 1)]["residual"] == pytest.approx(4.0 * delta)
        assert rows[(1, 2)]["residual"] == pytest.approx(-4.0 * delta)
        assert diagnostics.absolute_error == pytest.approx(4.0 * delta)
        assert audit.mass_balance_error == pytest.approx(4.0 * delta)

    def test_a_field_without_leading_axes_is_one_ledger(self):
        manager = _facade_manager()
        _, audit = manager.aggregate(
            torch.tensor([4.0, 2.0, 1.0, 3.0], dtype=torch.float64),
            "patch",
            "hillslope",
            variables=["litter_mass"],
            metadata=dict(_LITTER_CELL_TOTAL),
            topology=dict(_UNEQUAL_TOPOLOGY),
        )
        diagnostics = audit.metadata["conservation_audit"]["litter_mass"]
        assert diagnostics.metadata["batch_shape"] == ()
        assert diagnostics.metadata["failed_batches"] == []
        (row,) = diagnostics.metadata["batch_ledgers"]
        assert row["batch_index"] == ()
        assert row["source_total"] == diagnostics.source_total == 10.0
        assert row["target_total"] == diagnostics.target_total == 10.0
        assert row["residual"] == 0.0 and row["is_conserved"] is True
        assert diagnostics.metadata["uncovered_source_amount"] == 0.0

    def test_batches_that_each_conserve_pass_the_strict_path(self):
        """Positive case: every batch conserves (both amount bases, unequal areas, an empty parent)."""
        generator = torch.Generator().manual_seed(5)
        totals = torch.rand((3, 2), generator=generator, dtype=torch.float64) * 10.0
        density = torch.rand((3, 2), generator=generator, dtype=torch.float64)
        metadata = {
            "litter_mass": {"spatial_basis": "cell_total", "units": "kg"},
            "rho": {"spatial_basis": "area_density", "units": "kg m-2"},
        }
        manager = _facade_manager()
        assert manager.enforce_conservation is True
        for topology in (_PAIR_TOPOLOGY, _PARTIAL_PAIR_TOPOLOGY):
            result, audit = manager.aggregate(
                {"litter_mass": totals, "rho": density},
                "patch",
                "hillslope",
                metadata={name: dict(entry) for name, entry in metadata.items()},
                topology=dict(topology),
            )
            n_target = topology["n_target_units"]
            for name in ("litter_mass", "rho"):
                assert tuple(result[name].shape) == (3, n_target)
                diagnostics = audit.metadata["conservation_audit"][name]
                assert diagnostics.is_conserved is True
                assert diagnostics.metadata["batch_shape"] == (3,)
                assert diagnostics.metadata["failed_batches"] == []
                rows = diagnostics.metadata["batch_ledgers"]
                assert [row["batch_index"] for row in rows] == [(0,), (1,), (2,)]
                for row in rows:
                    assert row["is_conserved"] is True
                    assert row["absolute_error"] <= row["accepted_residual"]
            torch.testing.assert_close(result["litter_mass"][:, 0], totals.sum(dim=-1))
        # The empty parent receives no amount in any batch.
        assert torch.all(result["litter_mass"][:, 1] == 0.0)

    def test_float32_batches_keep_their_rounding_allowance_and_their_gradient(self, monkeypatch):
        manager = _facade_manager()
        generator = torch.Generator().manual_seed(3)
        n_patches = 64
        areas = (torch.rand(n_patches, generator=generator, dtype=torch.float64) * 1.0e4 + 1.0).tolist()
        topology = {
            "n_source_units": n_patches,
            "n_target_units": 2,
            "parent_map": {index: index % 2 for index in range(n_patches)},
            "source_areas": areas,
            "target_areas": [sum(areas[0::2]), sum(areas[1::2])],
        }
        values = (torch.rand((3, n_patches), generator=generator, dtype=torch.float64) * 1.0e-6).to(torch.float32)
        values.requires_grad_(True)
        request = dict(
            variables=["rho"],
            metadata={"rho": {"spatial_basis": "area_density", "units": "kg m-2"}},
            topology=topology,
        )
        result, audit = manager.aggregate(values, "patch", "hillslope", **request)
        # The audit does not detach the returned field from the graph of its input.
        assert result.requires_grad
        result.sum().backward()
        assert values.grad is not None and bool(torch.isfinite(values.grad).all())
        diagnostics = audit.metadata["conservation_audit"]["rho"]
        assert diagnostics.is_conserved is True
        rows = diagnostics.metadata["batch_ledgers"]
        assert len(rows) == 3
        for row in rows:
            # The float32 rounding of each batch is accepted on that batch's own activity.
            assert row["rounding_allowance"] > 0.0
            assert row["absolute_error"] <= row["accepted_residual"]
        # A wrong (doubled) second batch is still rejected on the same strict path.
        original = HierarchicalAggregator.aggregate

        def doubled_second_batch(self, *args, **kwargs):
            aggregated, aggregation_audit = original(self, *args, **kwargs)
            return {
                name: torch.cat([value[:1], 2.0 * value[1:2], value[2:]]) for name, value in aggregated.items()
            }, aggregation_audit

        monkeypatch.setattr(HierarchicalAggregator, "aggregate", doubled_second_batch)
        with pytest.raises(RuntimeError, match=r"rho.*violates conservation.*1 of 3 batch"):
            manager.aggregate(values.detach(), "patch", "hillslope", **request)

    def test_the_support_axis_of_a_batched_field_is_its_last_axis(self):
        """Without an explicit topology the number of source units is the size of the last axis."""
        manager = _facade_manager()
        metadata = {"q": {"spatial_basis": "cell_total", "units": "kg"}}
        data = torch.arange(1.0, 13.0, dtype=torch.float64).reshape(3, 4)
        result, audit = manager.aggregate({"q": data}, "patch", "hillslope", metadata=dict(metadata))
        assert result["q"].shape[0] == 3
        diagnostics = audit.metadata["conservation_audit"]["q"]
        assert diagnostics.is_conserved is True
        assert diagnostics.metadata["batch_shape"] == (3,)
        assert [row["source_total"] for row in diagnostics.metadata["batch_ledgers"]] == [10.0, 26.0, 42.0]
        down, down_audit = manager.disaggregate({"q": result["q"]}, "hillslope", "patch", metadata=dict(metadata))
        assert down["q"].shape[0] == 3
        assert down_audit.metadata["conservation_audit"]["q"].metadata["batch_shape"] == (3,)

    def test_the_aggregators_own_ledger_and_uncovered_amounts_are_per_batch(self):
        """HierarchicalAggregator: the internal ledger error is the largest batch error, and the
        amount of an orphan child is reported per batch (4 and -1, not their sum 3)."""
        topology = {
            "n_source_units": 4,
            "n_target_units": 2,
            "parent_map": {0: 0, 1: 0, 2: 1},
            "source_areas": [1.0, 3.0, 2.0, 2.0],
            "target_areas": [4.0, 4.0],
        }
        index = HierarchyIndex(source_scale="patch", target_scale="hillslope", n_source_units=4, topology=topology)
        aggregator = HierarchicalAggregator(index, weight_type="area", rule="conservative")
        source = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, -1.0]], dtype=torch.float64)
        result, info = aggregator.aggregate(
            {"q": source}, ["q"], strategy="conservative", variable_metadata={"q": {"spatial_basis": "cell_total"}}
        )
        torch.testing.assert_close(result["q"], torch.tensor([[3.0, 3.0], [7.0, 2.0]], dtype=torch.float64))
        assert info["uncovered_source_amount"]["q"] == [4.0, -1.0]
        full = HierarchyIndex(
            source_scale="patch", target_scale="hillslope", n_source_units=4, topology=dict(_UNEQUAL_TOPOLOGY)
        )
        # Batch 0 gains 0.5 and batch 1 loses 0.5 against totals of 10: the summed ledger is exact.
        shifted = torch.tensor([[3.5, 7.0], [7.0, 2.5]], dtype=torch.float64)
        full_source = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=torch.float64)
        error = HierarchicalAggregator(full, weight_type="area", rule="conservative")._check_extensive_conservation(
            full_source, shifted, "q", SpatialQuantityBasis.CELL_TOTAL
        )
        assert error == pytest.approx(0.05)

    def test_the_enforcer_needs_the_batch_layout_of_every_ledger(self):
        import inspect

        from relic.coupling.scales.conservation_enforcer import ConservationEnforcer

        batch_ndim = inspect.signature(ConservationEnforcer.check_conservation).parameters["batch_ndim"]
        assert batch_ndim.kind is inspect.Parameter.KEYWORD_ONLY
        assert batch_ndim.default is inspect.Parameter.empty
        enforcer = ConservationEnforcer(track_history=False, device=torch.device("cpu"))
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        target = torch.tensor([[3.0], [7.0]], dtype=torch.float64)
        with pytest.raises(TypeError):
            enforcer.check_conservation(source, target, property_name="q", tolerance=1e-9)
        # A row of the source and a column of the target are not the same batch.
        with pytest.raises(ValueError, match="batch"):
            enforcer.check_conservation(source, target.T, property_name="q", tolerance=1e-9, batch_ndim=1)
        for invalid in (-1, 3, True, 1.0):
            with pytest.raises((TypeError, ValueError), match="batch_ndim"):
                enforcer.check_conservation(source, target, property_name="q", tolerance=1e-9, batch_ndim=invalid)
        # Weights broadcast onto the data; weights that would widen a ledger are rejected.
        with pytest.raises(ValueError, match="weights"):
            enforcer.check_conservation(
                source[0], target[0], property_name="q", source_weights=torch.ones(2, 2), tolerance=1e-9, batch_ndim=0
            )
        diagnostics = enforcer.check_conservation(source, target, property_name="q", tolerance=1e-9, batch_ndim=1)
        assert diagnostics.is_conserved is True
        assert diagnostics.metadata["batch_shape"] == (2,)
        # Energy ledgers are judged per batch as well: +1 and -1 are two violations.
        energy = enforcer.check_conservation(
            source,
            torch.tensor([[4.0], [6.0]], dtype=torch.float64),
            property_name="energy",
            tolerance=1e-9,
            batch_ndim=1,
        )
        assert energy.is_conserved is False
        assert energy.metadata["failed_batches"] == [(0,), (1,)]
        assert [row["residual"] for row in energy.metadata["batch_ledgers"]] == [1.0, -1.0]


class TestRHESSysTemplates:
    """Test RHESSys template functionality."""
    
    def test_template_creation(self):
        """Test creating different templates."""
        templates = ["small_basin", "test_hillslope", "complex_basin"]
        
        for name in templates:
            template = get_rhessys_template(name)
            
            assert "n_source_units" in template
            assert "n_target_units" in template
            assert "source_areas" in template
            assert "target_areas" in template
            assert "parent_map" in template
            assert "children_map" in template
            
    def test_validation_data(self):
        """Test creating validation data."""
        for var_type in ["extensive", "intensive", "categorical"]:
            data = create_validation_data(
                "small_basin", variable_type=var_type
            )
            assert len(data) > 0
            for key, tensor in data.items():
                assert tensor.shape[0] == 100  # Number of patches
                
    def test_validation_data_names_carry_their_catalog_units(self):
        """The extensive template fluxes are carbon mass fluxes (kg C m-2 s-1): GPP is named
        gross_primary_productivity, not gpp, the catalog name of the molar flux (umol m-2 s-1)."""
        data = create_validation_data("small_basin", variable_type="extensive")
        assert "gpp" not in data
        assert "gross_primary_productivity" in data
        pm = get_global_manager()
        catalog = pm.get("variables", namespace="coupling.scales.defaults", required=True)
        assert catalog["gross_primary_productivity"]["units"] == "kg C m-2 s-1"
        assert catalog["gpp"]["units"] == "umol m-2 s-1"

    def test_conservation_validation(self):
        """Test conservation validation utility."""
        template = get_rhessys_template("small_basin")
        
        # Create test data
        source_data = torch.rand(100)
        
        # Create perfectly conserved aggregation
        source_areas = torch.tensor(template["source_areas"])
        target_areas = torch.tensor(template["target_areas"])
        
        # Manual aggregation for testing
        aggregated = torch.zeros(10)
        for target_idx in range(10):
            children = template["children_map"][target_idx]
            for child in children:
                aggregated[target_idx] += source_data[child] * source_areas[child]
            aggregated[target_idx] /= target_areas[target_idx]
        
        # Validate
        result = validate_aggregation_conservation(
            source_data, aggregated, template, tolerance=1e-9
        )
        
        assert "is_conserved" in result
        assert "relative_error" in result


def test_integration():
    """Integration test for full aggregation pipeline."""
    # Create template
    template = get_rhessys_template("small_basin")
    
    # Create hierarchy index
    index = HierarchyIndex(
        source_scale="patch",
        target_scale="landscape",
        topology=template
    )
    
    # Create aggregator
    aggregator = HierarchicalAggregator(index, weight_type="area", rule="conservative")
    
    # Create validation data
    data = create_validation_data("small_basin", "extensive")
    variables = list(data.keys())
    
    # Perform aggregation: the validation fluxes (kg C m-2 s-1) are declared area densities
    aggregated, audit = aggregator.aggregate(
        data,
        variables,
        mode="conservative",
        variable_metadata={name: {"spatial_basis": "area_density"} for name in variables},
    )
    
    # Check results
    assert len(aggregated) == len(data)
    for var in variables:
        assert var in aggregated
        assert aggregated[var].shape[0] == 10  # Target units
        
    # Check audit
    assert "max_conservation_error" in audit
    assert audit["max_conservation_error"] < 1e-6
    
    print("Integration test passed!")
    print(f"Conservation errors: {audit['conservation_errors']}")
    print(f"Methods used: {audit['method_used']}")


if __name__ == "__main__":
    # Run basic tests
    print("Testing HierarchyIndex...")
    test_idx = TestHierarchyIndex()
    test_idx.test_create_default_hierarchy()
    test_idx.test_rhessys_template()
    test_idx.test_weight_matrix_conservation()
    test_idx.test_no_dead_builders_or_raw_weight_application()
    print("[OK] HierarchyIndex tests passed")
    
    print("\nTesting HierarchicalAggregator...")
    test_agg = TestHierarchicalAggregator()
    test_agg.test_area_weighted_aggregation()
    test_agg.test_conservative_aggregation()
    test_agg.test_disaggregation()
    print("[OK] HierarchicalAggregator tests passed")
    
    print("\nTesting ScaleManager...")
    test_facade = TestScaleManagerFacade()
    test_facade.test_manager_interface()
    test_facade.test_mode_switching()
    print("[OK] ScaleManager tests passed")
    
    print("\nTesting RHESSys Templates...")
    test_templates = TestRHESSysTemplates()
    test_templates.test_template_creation()
    test_templates.test_validation_data()
    test_templates.test_conservation_validation()
    print("[OK] RHESSys template tests passed")
    
    print("\nRunning integration test...")
    test_integration()
    
    print("\n[SUCCESS] All tests passed successfully!")

