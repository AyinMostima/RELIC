"""
L1 Unit Tests for Spatial Aggregation Module

This module provides comprehensive unit tests for the SpatialAggregation class
and related spatial aggregation functionality, ensuring proper aggregation,
conservation properties, and different strategies.

Test Coverage:
- SpatialAggregation class initialization and configuration
- Forward method with different aggregation strategies
- Upscaling and downscaling methods
- Dependency management
- Error handling and edge cases

Mathematical Verification:
- Conservative aggregation: 危_target = 危_source * correction_factor
- Mean aggregation: 蠁_coarse = 危 (蠁_fine * weights) / 危 weights
- Sum aggregation: 蠁_coarse = 危 蠁_fine

References:
- Jones, P.W. (1999). Conservative remapping schemes for climate models
- CF Conventions 1.8: Cell methods for spatial aggregation
"""

import pytest
import torch

# Import RELIC modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from relic.coupling.scales.adapters import SpatialAggregation


class TestSpatialAggregationInit:
    """Test SpatialAggregation class initialization and configuration."""

    def test_init_basic(self):
        """Test basic SpatialAggregation initialization."""
        aggregation = SpatialAggregation(
            aggregation_strategy="mean",
            preserve_totals=True,
            weighting_type="area",
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        assert aggregation.aggregation_strategy == "mean"
        assert aggregation.preserve_totals is True
        assert aggregation.weighting_type == "area"
        assert aggregation.source_scale == "fine_grid"
        assert aggregation.target_scale == "coarse_grid"
        assert aggregation.coupling_type == "spatial"

    def test_init_different_strategies(self):
        """Every selector that names a basis law is accepted (MA-21 item 122)."""
        strategies = [
            "area_weighted", "mean", "sum", "total", "conservative",
            "area_weighted_mean", "area_weighted_sum",
        ]

        for strategy in strategies:
            aggregation = SpatialAggregation(
                aggregation_strategy=strategy,
                preserve_totals=True,
                source_scale="fine_grid",
                target_scale="coarse_grid",
            )
            assert aggregation.aggregation_strategy == strategy

    @pytest.mark.parametrize("strategy", ["max", "min", "median"])
    def test_selectors_without_a_basis_law_are_rejected(self, strategy):
        """max, min and median were accepted and silently ignored; no basis law defines them."""
        with pytest.raises(ValueError, match="aggregation_strategy"):
            SpatialAggregation(aggregation_strategy=strategy, preserve_totals=True, source_scale="fine_grid", target_scale="coarse_grid")

    def test_weighting_type_vocabulary_is_the_accepted_one(self):
        """weighting_type names the measure of the weights; only area measures define the law."""
        from relic.parameters import get_global_manager

        spec = get_global_manager().get_parameter("coupling.scales.spatial.weighting_type")
        assert list(spec.allowed_values) == ["area"]
        for value in ("mass", "volume", "uniform"):
            with pytest.raises(ValueError, match="weighting_type"):
                SpatialAggregation(aggregation_strategy="area_weighted", weighting_type=value, source_scale="fine_grid", target_scale="coarse_grid")

    def test_init_conservative_mode(self):
        """Test initialization in conservative mode."""
        aggregation = SpatialAggregation(
            aggregation_strategy="sum",
            preserve_totals=True,
            weighting_type="area",
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        assert "mass" in aggregation.conservation_properties
        assert "energy" in aggregation.conservation_properties

    def test_init_device_specification(self):
        """Test initialization with device specification."""
        device = torch.device("cpu")
        aggregation = SpatialAggregation(
            aggregation_strategy="mean",
            preserve_totals=True,
            device=device,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        assert aggregation.device == device


class TestSpatialAggregationForward:
    """Test SpatialAggregation forward method."""

    def test_forward_mean_aggregation(self):
        """Test forward method with mean aggregation."""
        aggregation = SpatialAggregation(
            aggregation_strategy="mean",
            preserve_totals=False,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        # Create test state with spatial data
        state = {
            'test_field': torch.tensor([[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0]
            ]], dtype=torch.float32)
        }

        time = 0.0
        dt = 3600.0

        result = aggregation.forward(
            state, time, dt, variable_metadata={"test_field": {"spatial_basis": "intensive"}}
        )

        # Unweighted mean of 1..16 over the whole support.
        assert float(result['test_field_aggregated']) == pytest.approx(8.5)

    def test_forward_area_density_keeps_its_basis(self):
        """An area-density catalog variable is averaged with its cell areas; its ledger is 1.8e-7."""
        aggregation = SpatialAggregation(
            aggregation_strategy="area_weighted",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        # net_primary_productivity (kg C m-2 s-1 per unit ground area, contracted) is an area
        # density in coupling.scales.defaults.variables.
        state = {
            'net_primary_productivity': torch.ones(1, 3, 3, dtype=torch.float32) * 2.0e-8
        }
        areas = torch.ones(1, 3, 3, dtype=torch.float32)

        result = aggregation.forward(state, 0.0, 1800.0, weights=areas)

        # rho = sum_i x_i A_i / sum_i A_i = 2e-8 (a density stays a density); amount 2e-8 * 9 = 1.8e-7.
        assert float(result['net_primary_productivity_aggregated']) == pytest.approx(2.0e-8)
        audit = aggregation.last_scaling_audit
        assert audit.metadata['output_basis'] == {'net_primary_productivity': 'area_density'}
        ledger = audit.metadata['conservation_audit']['net_primary_productivity']
        assert ledger.source_total == pytest.approx(1.8e-7)
        assert ledger.target_total == pytest.approx(1.8e-7)
        assert audit.conservation_enforced is True

    def test_a_sum_selector_conflicts_with_an_area_density(self):
        """'sum' names the cell-total law; an area density is not summed without its areas."""
        aggregation = SpatialAggregation(aggregation_strategy="sum", preserve_totals=True, source_scale="fine_grid", target_scale="coarse_grid")
        with pytest.raises(ValueError, match="conflicts with the declared basis"):
            aggregation.forward(
                {'net_primary_productivity': torch.ones(1, 3, 3) * 2.0e-8}, 0.0, 1800.0, weights=torch.ones(1, 3, 3)
            )

    def test_a_mean_selector_conflicts_with_a_cell_total(self):
        """'mean' names the density and intensive law; cell totals are summed."""
        aggregation = SpatialAggregation(aggregation_strategy="mean", preserve_totals=False, source_scale="fine_grid", target_scale="coarse_grid")
        with pytest.raises(ValueError, match="conflicts with the declared basis"):
            aggregation.forward(
                {'pool': torch.tensor([[4.0, 8.0]], dtype=torch.float64)}, 0.0, 1.0,
                variable_metadata={'pool': {'spatial_basis': 'cell_total'}},
            )

    def test_forward_with_custom_variables(self):
        """Test forward method with specific variables."""
        aggregation = SpatialAggregation(
            aggregation_strategy="mean",
            preserve_totals=False,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        # Create test state with multiple fields
        state = {
            'temperature': torch.ones(1, 4, 4, dtype=torch.float32) * 300.0,
            'humidity': torch.ones(1, 4, 4, dtype=torch.float32) * 0.5,
            'pressure': torch.ones(1, 4, 4, dtype=torch.float32) * 101325.0
        }

        time = 0.0
        dt = 3600.0

        # Only aggregate specific variables
        result = aggregation.forward(
            state, time, dt, variables=['temperature', 'humidity'],
            variable_metadata={
                'temperature': {'spatial_basis': 'intensive'},
                'humidity': {'spatial_basis': 'intensive'},
            },
        )

        assert set(result) == {'temperature_aggregated', 'humidity_aggregated'}
        assert float(result['temperature_aggregated']) == pytest.approx(300.0)
        assert float(result['humidity_aggregated']) == pytest.approx(0.5)

    def test_forward_with_empty_state(self):
        """Test forward method with empty state."""
        aggregation = SpatialAggregation(
            aggregation_strategy="mean",
            preserve_totals=False,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        state = {}
        time = 0.0
        dt = 3600.0

        result = aggregation.forward(state, time, dt)

        # Should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0

class TestSpatialAggregationScaling:
    """Test SpatialAggregation scaling methods."""

    def test_upscale_method(self):
        """upscale reduces one declared field through forward."""
        aggregation = SpatialAggregation(
            aggregation_strategy="mean",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        fine_data = torch.ones(1, 4, 4, dtype=torch.float32) * 2.0

        result = aggregation.upscale(
            fine_data, variable_metadata={"data": {"spatial_basis": "intensive"}}
        )
        assert float(result) == pytest.approx(2.0)

    def test_downscale_method(self):
        """SpatialAggregation has no disaggregation and says so instead of returning the input."""
        aggregation = SpatialAggregation(
            aggregation_strategy="mean",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        coarse_data = torch.ones(1, 2, 2, dtype=torch.float32) * 4.0

        with pytest.raises(NotImplementedError, match="downscale"):
            aggregation.downscale(coarse_data)


class TestSpatialAggregationDependencies:
    """Test SpatialAggregation dependency management."""

    def test_get_dependencies(self):
        """Test dependency declaration."""
        aggregation = SpatialAggregation(
            aggregation_strategy="mean",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        dependencies = aggregation.get_dependencies()
        # Should return empty list as spatial aggregation has no dependencies
        assert dependencies == []


class TestErrorHandling:
    """Test error handling for spatial aggregation."""

    def test_invalid_strategy(self):
        """Test initialization with invalid strategy."""
        with pytest.raises(ValueError):
            SpatialAggregation(
                aggregation_strategy="invalid_strategy",
                preserve_totals=True,
                source_scale="fine_grid",
                target_scale="coarse_grid",
            )

    def test_forward_with_device_mismatch(self):
        """Test forward method with device mismatch."""
        device = torch.device("cpu")
        aggregation = SpatialAggregation(
            aggregation_strategy="mean",
            preserve_totals=False,
            device=device,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        # Create data on same device
        state = {
            'test_field': torch.ones(1, 2, 2, dtype=torch.float32, device=device) * 5.0
        }

        time = 0.0
        dt = 3600.0

        result = aggregation.forward(
            state, time, dt, variable_metadata={"test_field": {"spatial_basis": "intensive"}}
        )

        # Should handle device consistency
        assert isinstance(result, dict)
        assert result['test_field_aggregated'].device == device
        assert float(result['test_field_aggregated']) == pytest.approx(5.0)


class TestExplicitSpatialBasis:
    """The declared SpatialQuantityBasis, not the variable name, selects the reduction."""

    @staticmethod
    def _aggregation(preserve_totals: bool = False) -> SpatialAggregation:
        return SpatialAggregation(
            aggregation_strategy="area_weighted",
            preserve_totals=preserve_totals,
            weighting_type="area",
            device=torch.device("cpu"),
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

    def test_basis_selects_the_mean_or_the_total(self):
        aggregation = self._aggregation()
        areas = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
        state = {
            "canopy_temperature": torch.tensor([[280.0, 300.0]], dtype=torch.float64),
            "carbon_ratio_pool": torch.tensor([[4.0, 8.0]], dtype=torch.float64),
            "soil_carbon_density": torch.tensor([[4.0, 8.0]], dtype=torch.float64),
        }
        result = aggregation.forward(
            state, 0.0, 1.0,
            weights=areas,
            variable_metadata={
                "canopy_temperature": {"spatial_basis": "intensive"},
                "carbon_ratio_pool": {"spatial_basis": "cell_total"},
                "soil_carbon_density": {"spatial_basis": "area_density"},
            },
        )
        # intensive: area-weighted mean (280*1 + 300*3) / 4
        assert float(result["canopy_temperature_aggregated"]) == pytest.approx(295.0)
        # cell totals are summed without an area factor, whatever the name says
        assert float(result["carbon_ratio_pool_aggregated"]) == pytest.approx(12.0)
        # area densities are averaged with the cell areas: (4*1 + 8*3) / 4, ledger 28
        assert float(result["soil_carbon_density_aggregated"]) == pytest.approx(7.0)
        audit = aggregation.last_scaling_audit
        assert audit.metadata["output_basis"] == {
            "canopy_temperature": "intensive",
            "carbon_ratio_pool": "cell_total",
            "soil_carbon_density": "area_density",
        }
        assert audit.metadata["quantity_basis"] == audit.metadata["output_basis"]
        assert audit.metadata["conservation_audited_variables"] == [
            "carbon_ratio_pool", "soil_carbon_density",
        ]
        ledger = audit.metadata["conservation_audit"]["soil_carbon_density"]
        assert ledger.source_total == pytest.approx(28.0)
        assert ledger.target_total == pytest.approx(28.0)
        assert not hasattr(aggregation, "last_scaling_audits")

    def test_categorical_fields_take_the_area_weighted_mode(self):
        aggregation = self._aggregation()
        result = aggregation.forward(
            {"land_cover": torch.tensor([[3.0, 7.0]], dtype=torch.float64)}, 0.0, 1.0,
            weights=torch.tensor([[1.0, 3.0]], dtype=torch.float64),
            variable_metadata={"land_cover": {"spatial_basis": "categorical"}},
        )
        # class 7 covers 3 of the 4 m2.
        assert float(result["land_cover_aggregated"]) == pytest.approx(7.0)

    def test_preserve_totals_enforces_the_ledger_of_the_returned_value(self, monkeypatch):
        """A wrong result raises with preserve_totals and is reported (never zero) without it."""
        import relic.coupling.scales.alias_utils as alias_utils

        original = alias_utils._reduce_on_basis
        monkeypatch.setattr(alias_utils, "_reduce_on_basis", lambda *a, **k: 2.0 * original(*a, **k))
        state = {"soil_carbon_density": torch.tensor([[4.0, 8.0]], dtype=torch.float64)}
        request = dict(
            weights=torch.tensor([[1.0, 3.0]], dtype=torch.float64),
            variable_metadata={"soil_carbon_density": {"spatial_basis": "area_density"}},
        )
        with pytest.raises(RuntimeError, match="violates conservation"):
            self._aggregation(preserve_totals=True).forward(state, 0.0, 1.0, **request)
        relaxed = self._aggregation(preserve_totals=False)
        result = relaxed.forward(state, 0.0, 1.0, **request)
        assert float(result["soil_carbon_density_aggregated"]) == pytest.approx(14.0)
        assert relaxed.last_scaling_audit.relative_error == pytest.approx(1.0)
        assert relaxed.last_scaling_audit.conservation_enforced is False

    def test_preserve_totals_does_not_sum_intensive_fields(self):
        aggregation = self._aggregation(preserve_totals=True)
        result = aggregation.forward(
            {"canopy_temperature": torch.tensor([[280.0, 300.0]], dtype=torch.float64)}, 0.0, 1.0,
            variable_metadata={"canopy_temperature": {"spatial_basis": "intensive"}},
        )
        assert float(result["canopy_temperature_aggregated"]) == pytest.approx(290.0)

    def test_area_density_without_cell_areas_is_rejected(self):
        aggregation = self._aggregation()
        with pytest.raises(ValueError, match="cell areas"):
            aggregation.forward(
                {"soil_carbon_density": torch.tensor([[4.0, 8.0]], dtype=torch.float64)}, 0.0, 1.0,
                variable_metadata={"soil_carbon_density": {"spatial_basis": "area_density"}},
            )

    def test_undeclared_or_bare_extensive_variables_are_rejected(self):
        aggregation = self._aggregation()
        state = {"test_field": torch.ones(1, 2, 2, dtype=torch.float64)}
        with pytest.raises(ValueError, match="declare spatial_basis"):
            aggregation.forward(state, 0.0, 1.0)
        with pytest.raises(ValueError, match="ambiguous"):
            aggregation.forward(
                state, 0.0, 1.0, variable_metadata={"test_field": {"variable_type": "extensive"}}
            )

    def test_mismatched_weights_are_rejected(self):
        aggregation = self._aggregation()
        with pytest.raises(ValueError, match="shape"):
            aggregation.forward(
                {"canopy_temperature": torch.ones(1, 2, 2, dtype=torch.float64)}, 0.0, 1.0,
                weights=torch.ones(4, dtype=torch.float64),
                variable_metadata={"canopy_temperature": {"spatial_basis": "intensive"}},
            )

    def test_rank_one_and_rank_zero_fields_are_aggregated_not_skipped(self):
        aggregation = self._aggregation()
        result = aggregation.forward(
            {
                "patch_temperature": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
                "single_cell_temperature": torch.tensor(5.0, dtype=torch.float64),
            },
            0.0, 1.0,
            variable_metadata={
                "patch_temperature": {"spatial_basis": "intensive"},
                "single_cell_temperature": {"spatial_basis": "intensive"},
            },
        )
        assert float(result["patch_temperature_aggregated"]) == pytest.approx(2.0)
        assert float(result["single_cell_temperature_aggregated"]) == pytest.approx(5.0)

    def test_aggregates_are_not_clamped_by_variable_name(self):
        aggregation = self._aggregation()
        result = aggregation.forward(
            {
                "soil_moisture_volume": torch.tensor([[0.5, 0.7]], dtype=torch.float64),
                "leaf_area_index": torch.tensor([[12.0, 14.0]], dtype=torch.float64),
            },
            0.0, 1.0,
            variable_metadata={
                "soil_moisture_volume": {"spatial_basis": "cell_total"},
                "leaf_area_index": {"spatial_basis": "intensive"},
            },
        )
        assert float(result["soil_moisture_volume_aggregated"]) == pytest.approx(1.2)
        assert float(result["leaf_area_index_aggregated"]) == pytest.approx(13.0)

    def test_governed_call_runs_the_same_forward(self):
        aggregation = self._aggregation()
        result = aggregation(
            {"canopy_temperature": torch.tensor([[280.0, 300.0]], dtype=torch.float64)}, 0.0, 1.0,
            variables=["canopy_temperature"],
            variable_metadata={"canopy_temperature": {"spatial_basis": "intensive"}},
        )
        assert float(result["canopy_temperature_aggregated"]) == pytest.approx(290.0)


# Test fixtures
@pytest.fixture
def basic_aggregation():
    """Fixture providing basic spatial aggregation."""
    return SpatialAggregation(
        aggregation_strategy="mean",
        preserve_totals=True,
        weighting_type="area",
        source_scale="fine_grid",
        target_scale="coarse_grid",
    )


@pytest.fixture
def sample_spatial_data():
    """Fixture providing sample spatial data."""
    return {
        'temperature': torch.ones(1, 4, 4, dtype=torch.float32) * 300.0,
        'net_primary_productivity': torch.ones(1, 4, 4, dtype=torch.float32) * 2.0e-8
    }


# Integration tests using fixtures
def test_complete_aggregation_workflow(
    basic_aggregation,
    sample_spatial_data
):
    """Integration test for complete aggregation workflow."""
    time = 0.0
    dt = 3600.0

    # temperature is declared per call; net_primary_productivity is an area density in the YAML catalog.
    result = basic_aggregation.forward(
        sample_spatial_data, time, dt,
        weights=torch.ones(1, 4, 4, dtype=torch.float32),
        variable_metadata={"temperature": {"spatial_basis": "intensive"}},
    )

    assert set(result) == {'temperature_aggregated', 'net_primary_productivity_aggregated'}
    assert float(result['temperature_aggregated']) == pytest.approx(300.0)
    # The area density keeps its basis: the area-weighted mean on unit cell areas.
    assert float(result['net_primary_productivity_aggregated']) == pytest.approx(2.0e-8)


def test_different_strategies_comparison():
    """Integration test comparing different aggregation strategies."""
    state = {
        'test_field': torch.tensor([[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]], dtype=torch.float32)
    }

    metadata = {"test_field": {"spatial_basis": "intensive"}}
    for strategy in ("area_weighted", "mean", "conservative"):
        aggregation = SpatialAggregation(aggregation_strategy=strategy, preserve_totals=False, source_scale="fine_grid", target_scale="coarse_grid")
        result = aggregation.forward(state, 0.0, 3600.0, variable_metadata=metadata)
        assert float(result['test_field_aggregated']) == pytest.approx(8.5)

    # A selector outside the intensive law is a conflict, not a silently ignored label.
    for strategy in ("sum", "total"):
        aggregation = SpatialAggregation(aggregation_strategy=strategy, preserve_totals=False, source_scale="fine_grid", target_scale="coarse_grid")
        with pytest.raises(ValueError, match="conflicts with the declared basis"):
            aggregation.forward(state, 0.0, 3600.0, variable_metadata=metadata)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])


def test_spatial_aggregation_scales_are_required():
    """MA-21 item 205: SpatialAggregation (and the aggregate_spatial functional API) state the
    source and target scale they couple; the constructor keeps no fine_grid/coarse_grid default."""
    import inspect

    from relic.coupling.scales.adapters import SpatialAggregation, aggregate_spatial

    for function in (SpatialAggregation.__init__, aggregate_spatial):
        parameters = inspect.signature(function).parameters
        for name in ("source_scale", "target_scale"):
            assert parameters[name].default is inspect.Parameter.empty, (function, name)
    with pytest.raises(TypeError, match="source_scale"):
        SpatialAggregation(target_scale="coarse_grid")
    aggregator = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
    assert (aggregator.source_scale, aggregator.target_scale) == ("fine_grid", "coarse_grid")

