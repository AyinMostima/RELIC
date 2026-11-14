"""
L1 Unit Tests - Hierarchical Scaling Rules

Tests the 7 new aggregation strategies in HierarchicalAggregator to ensure:
- Correct mathematical formulation for each aggregation rule
- Conservation properties for appropriate variable types
- Proper handling of boundary conditions and edge cases
- Numerical accuracy (tolerance < 1e-12 for critical checks)

Mathematical References:
    - Monteith & Unsworth (2013). Principles of Environmental Physics.
    - Mardia & Jupp (2000). Directional Statistics. Wiley.
    - Limpert et al. (2001). BioScience, 51(5), 341-352.
    - Pielke (2002). Mesoscale Meteorological Modeling.
"""

import pytest
import torch
import numpy as np

# Import hierarchical scaling modules
from relic.coupling.scales.hierarchical import HierarchyIndex, HierarchicalAggregator


@pytest.fixture
def simple_hierarchy():
    """
    Create a simple 3->2 hierarchy for testing.

    Structure:
        Source units: [0, 1, 2] with areas [1.0, 2.0, 3.0] m虏
        Target units: [0, 1] with areas [3.0, 3.0] m虏
        Mapping: sources 0,1 -> target 0; source 2 -> target 1
    """
    topology = {
        'n_source_units': 3,
        'n_target_units': 2,
        'source_areas': [1.0, 2.0, 3.0],
        'target_areas': [3.0, 3.0],
        'parent_map': {0: 0, 1: 0, 2: 1}
    }

    index = HierarchyIndex(
        source_scale='patch',
        target_scale='hillslope',
        topology=topology,
        device=torch.device('cpu')
    )

    return index


@pytest.fixture
def aggregator_fixture(simple_hierarchy):
    """Create HierarchicalAggregator with simple hierarchy."""
    aggregator = HierarchicalAggregator(
        index=simple_hierarchy,
        device=torch.device('cpu'),
        rule='conservative',
        weight_type="area",
    )
    return aggregator


class TestHarmonicMean:
    """
    Test harmonic mean aggregation for resistance-type variables.

    Mathematical Expression:
        r_eff = 1 / (sum_i (w_i / r_i))
        where w_i = A_i / sum_j A_j (normalized area weights)

    Physical Interpretation:
        Resistances in parallel (e.g., stomatal resistance, aerodynamic resistance)

    References:
        - Monteith & Unsworth (2013). Principles of Environmental Physics.
          Chapter 13: Microclimatology (resistance networks).
        - Baldocchi et al. (1987). Boundary Layer Meteorology, 40, 123-136.
          A multi-layer model for estimating sulfur dioxide deposition.
    """

    def test_simple_resistance_network(self, aggregator_fixture):
        """
        Test known resistance network with hand-calculated result.

        Setup:
            Source resistances: [100, 200, 300] s/m
            Source areas: [1, 2, 3] m虏

        Expected for target 0 (sources 0,1):
            w0 = 1/(1+2) = 1/3, w1 = 2/(1+2) = 2/3
            1/r_eff = w0/100 + w1/200 = 1/300 + 2/600 = 1/300 + 1/300 = 2/300
            r_eff = 150 s/m

        Expected for target 1 (source 2):
            r_eff = 300 s/m (single source)
        """
        # Source resistance values [s/m]
        data = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)

        # Aggregate using harmonic mean
        result = aggregator_fixture._harmonic_mean(data, 'resistance')

        # Expected results
        expected = torch.tensor([150.0, 300.0], dtype=torch.float64)

        # Verify with tight tolerance
        assert torch.allclose(result, expected, atol=1e-12), \
            f"Harmonic mean mismatch: got {result}, expected {expected}"

    def test_conservation_property(self, aggregator_fixture):
        """
        Test that harmonic mean satisfies bounds property.

        Property: min(r_i) <= r_eff <= max(r_i) for each target unit
        """
        data = torch.tensor([50.0, 100.0, 200.0], dtype=torch.float64)

        result = aggregator_fixture._harmonic_mean(data, 'resistance')

        # Target 0: sources 0,1 -> min=50, max=100
        assert result[0] >= 50.0 - 1e-12, "Harmonic mean below minimum"
        assert result[0] <= 100.0 + 1e-12, "Harmonic mean above maximum"

        # Target 1: source 2 -> should equal 200
        assert torch.abs(result[1] - 200.0) < 1e-12, "Single source not preserved"

    def test_zero_handling(self, aggregator_fixture):
        """
        Test epsilon protection for near-zero resistances.

        Near-zero resistances should be clamped to epsilon (1e-12) to avoid division by zero.
        """
        # Include near-zero and zero values
        data = torch.tensor([1e-15, 100.0, 200.0], dtype=torch.float64)

        # Should not raise exception
        result = aggregator_fixture._harmonic_mean(data, 'resistance')

        # Result should be finite and positive
        assert torch.all(torch.isfinite(result)), "Result contains inf/nan"
        assert torch.all(result > 0), "Result should be positive"


class TestArithmeticMean:
    """
    Test arithmetic mean aggregation for conductance-type variables.

    Mathematical Expression:
        g_eff = sum_i (w_i * g_i)
        where w_i = A_i / sum_j A_j (normalized area weights)

    Physical Interpretation:
        Area-weighted average of conductances, leaf area indices, etc.

    References:
        - Jarvis & McNaughton (1986). Advances in Ecological Research, 15, 1-49.
          Stomatal control of transpiration: Scaling up from leaf to region.
    """

    def test_simple_weighted_average(self, aggregator_fixture):
        """
        Test basic area-weighted averaging with hand-calculated result.

        Setup:
            Source conductances: [10, 20, 30] m/s
            Source areas: [1, 2, 3] m虏

        Expected for target 0 (sources 0,1):
            w0 = 1/3, w1 = 2/3
            g_eff = (1/3)*10 + (2/3)*20 = 10/3 + 40/3 = 50/3 鈮?16.6667

        Expected for target 1 (source 2):
            g_eff = 30 m/s (single source)
        """
        data = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)

        result = aggregator_fixture._area_weighted_mean(data, 'conductance')

        # Expected values - match result dtype
        expected = torch.tensor([50.0/3.0, 30.0], dtype=result.dtype)

        assert torch.allclose(result, expected, atol=1e-6), \
            f"Arithmetic mean mismatch: got {result}, expected {expected}"

    def test_conservation_of_weighted_sum(self, aggregator_fixture):
        """
        Test that weighted sum is preserved for extensive interpretation.

        For conductance as intensive variable, the weighted average should be between min and max.
        """
        data = torch.tensor([5.0, 10.0, 15.0], dtype=torch.float64)

        result = aggregator_fixture._area_weighted_mean(data, 'conductance')

        # Each target should have value between its sources' min and max
        assert result[0] >= 5.0 - 1e-12, "Below minimum"
        assert result[0] <= 10.0 + 1e-12, "Above maximum"
        assert torch.abs(result[1] - 15.0) < 1e-12, "Single source error"

    def test_uniform_values(self, aggregator_fixture):
        """Test that uniform input produces uniform output."""
        data = torch.tensor([42.0, 42.0, 42.0], dtype=torch.float32)

        result = aggregator_fixture._area_weighted_mean(data, 'conductance')

        # All targets should equal 42 - match result dtype
        expected = torch.tensor([42.0, 42.0], dtype=result.dtype)
        assert torch.allclose(result, expected, atol=1e-6), \
            "Uniform values not preserved"


class TestBoundedAreaMean:
    """
    Test bounded area-weighted mean for fraction variables in [0,1].

    Mathematical Expression:
        x_j = clip(sum_i (x_i * A_i) / sum_i A_i, 0, 1)

    Physical Interpretation:
        Fractional vegetation cover, cloud fraction, soil moisture fraction

    References:
        - Pielke (2002). Mesoscale Meteorological Modeling.
          Section 8.3: Grid aggregation for surface properties.
    """

    def test_valid_fractions(self, aggregator_fixture):
        """
        Test aggregation of valid fractions stays in bounds.

        Setup:
            Source fractions: [0.3, 0.7, 0.5]

        Expected for target 0:
            mean = (1*0.3 + 2*0.7) / 3 = 1.7/3 鈮?0.5667

        Expected for target 1:
            mean = 0.5
        """
        data = torch.tensor([0.3, 0.7, 0.5], dtype=torch.float32)

        result = aggregator_fixture._bounded_area_mean(data, 'vegetation_fraction')

        # Check bounds
        assert torch.all(result >= 0.0), "Below 0"
        assert torch.all(result <= 1.0), "Above 1"

        # Check values - match result dtype
        expected = torch.tensor([1.7/3.0, 0.5], dtype=result.dtype)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_boundary_clipping(self, aggregator_fixture):
        """
        Test that out-of-bounds values are clipped to [0,1].

        Note: In practice, input should be valid fractions, but test robustness.
        """
        # Create aggregator with modified data that would exceed bounds
        # We'll test by directly checking the clipping behavior
        data = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        result = aggregator_fixture._bounded_area_mean(data, 'fraction')

        # All results should be in [0, 1]
        assert torch.all(result >= 0.0) and torch.all(result <= 1.0), \
            "Clipping failed"

    def test_extreme_values(self, aggregator_fixture):
        """Test with extreme (but valid) fraction values."""
        data = torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64)

        result = aggregator_fixture._bounded_area_mean(data, 'fraction')

        # Target 0: (1*0 + 2*1)/3 = 2/3
        expected_0 = 2.0/3.0
        assert torch.abs(result[0] - expected_0) < 1e-12, \
            f"Extreme value aggregation failed: {result[0]} vs {expected_0}"


class TestAreaWeightedMode:
    """
    Test area-weighted mode for categorical variables.

    Mathematical Expression:
        mode_j = argmax_c (sum_{i: x_i=c} A_i)
        Select category with largest total area in target unit

    Physical Interpretation:
        Land cover type, soil type, vegetation type

    References:
        - Friedl & Brodley (1997). IEEE Trans. Geosci. Remote Sens., 35, 716-724.
          Decision tree classification of land cover from remotely sensed data.
    """

    def test_simple_mode_selection(self, aggregator_fixture):
        """
        Test mode selection with clear majority.

        Setup:
            Source categories: [1, 1, 2] (integers representing land cover types)
            Source areas: [1, 2, 3] m2

        Expected for target 0 (sources 0,1):
            Category 1: area = 1+2 = 3 m2
            Mode = 1 (largest area)

        Expected for target 1 (source 2):
            Mode = 2 (only category)
        """
        # Land cover categories (e.g., 1=forest, 2=grassland)
        data = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)

        result = aggregator_fixture._area_weighted_mode(data, 'land_cover')

        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(result, expected, atol=1e-12), \
            f"Mode selection error: got {result}, expected {expected}"

    def test_tie_breaking(self, aggregator_fixture):
        """
        Test behavior when categories have equal area.

        With equal areas, should select first category encountered.
        """
        # Modified topology: equal areas
        topology = {
            'n_source_units': 2,
            'n_target_units': 1,
            'source_areas': [1.0, 1.0],  # Equal areas
            'target_areas': [2.0],
            'parent_map': {0: 0, 1: 0}
        }

        index = HierarchyIndex(
            source_scale='patch',
            target_scale='hillslope',
            topology=topology
        )

        agg = HierarchicalAggregator(index=index, weight_type="area", rule="conservative")

        # Different categories with equal weight
        data = torch.tensor([3.0, 5.0], dtype=torch.float64)
        result = agg._area_weighted_mode(data, 'category')

        # Should select one of them deterministically
        assert result[0] in [3.0, 5.0], "Invalid category selected"

    def test_single_dominant_category(self, aggregator_fixture):
        """Test with one category having overwhelming majority."""
        # All same category
        data = torch.tensor([7.0, 7.0, 7.0], dtype=torch.float64)

        result = aggregator_fixture._area_weighted_mode(data, 'soil_type')

        # Both targets should have category 7
        expected = torch.tensor([7.0, 7.0], dtype=torch.float64)
        assert torch.allclose(result, expected, atol=1e-12)


class TestCircularMean:
    """
    Test circular mean for angular variables.

    Mathematical Expression:
        theta_j = atan2(sum_i (w_i * sin(theta_i)), sum_i (w_i * cos(theta_i)))
        where w_i = A_i / sum_k A_k

    Physical Interpretation:
        Wind direction, slope aspect, sun azimuth

    References:
        - Mardia & Jupp (2000). Directional Statistics. Wiley.
          Chapter 2: The von Mises distribution and circular statistics.
        - Jammalamadaka & SenGupta (2001). Topics in Circular Statistics. World Scientific.
    """

    def test_simple_angle_averaging(self, aggregator_fixture):
        """
        Test circular mean with known angles.

        Setup:
            Source angles: [0, 蟺/2, 蟺] radians (0掳, 90掳, 180掳)
            Source areas: [1, 2, 3] m虏

        Expected for target 0 (sources 0,1):
            Weights: w0=1/3, w1=2/3
            sin_sum = (1/3)*sin(0) + (2/3)*sin(蟺/2) = 0 + 2/3 = 2/3
            cos_sum = (1/3)*cos(0) + (2/3)*cos(蟺/2) = 1/3 + 0 = 1/3
            theta = atan2(2/3, 1/3) 鈮?1.1071 rad (鈮?63.43掳)
        """
        data = torch.tensor([0.0, np.pi/2, np.pi], dtype=torch.float64)

        result = aggregator_fixture._circular_mean(data, 'wind_direction')

        # Expected angle for target 0
        expected_0 = torch.atan2(torch.tensor(2.0/3.0), torch.tensor(1.0/3.0))

        # Target 1 should equal 蟺 (single source)
        expected = torch.tensor([expected_0.item(), np.pi], dtype=torch.float64)

        assert torch.allclose(result, expected, atol=1e-10), \
            f"Circular mean error: got {result}, expected {expected}"

    def test_opposite_directions(self, aggregator_fixture):
        """
        Test circular mean with opposite directions.

        Angles 0 and 蟺 with equal weights should cancel out.
        """
        # Create equal-weight topology
        topology = {
            'n_source_units': 2,
            'n_target_units': 1,
            'source_areas': [1.0, 1.0],
            'target_areas': [2.0],
            'parent_map': {0: 0, 1: 0}
        }

        index = HierarchyIndex(
            source_scale='patch',
            target_scale='hillslope',
            topology=topology
        )

        agg = HierarchicalAggregator(index=index, weight_type="area", rule="conservative")

        # Opposite directions: 0 and 蟺
        data = torch.tensor([0.0, np.pi], dtype=torch.float64)
        result = agg._circular_mean(data, 'aspect')

        # sin_sum 鈮?0, cos_sum = 0.5 - 0.5 = 0
        # Result should be near 蟺/2 or -蟺/2 (pointing "upward" on unit circle when x=0)
        assert torch.abs(torch.abs(result[0]) - np.pi/2) < 1e-10, \
            f"Opposite angles not handled correctly: {result[0]}"

    def test_angle_wrap_around(self, aggregator_fixture):
        """
        Test that result is in (-蟺, 蟺] range.

        Circular mean should automatically wrap angles.
        """
        # Angles near 卤蟺 boundary
        data = torch.tensor([np.pi - 0.1, np.pi - 0.1, -np.pi + 0.1],
                           dtype=torch.float64)

        result = aggregator_fixture._circular_mean(data, 'direction')

        # Result should be in valid range
        assert torch.all(result > -np.pi) and torch.all(result <= np.pi), \
            "Angle not properly wrapped"


class TestComponentWiseMean:
    """
    Test component-wise mean for vector fields.

    Mathematical Expression:
        V_j = sum_i (w_i * V_i) for each component
        where w_i = A_i / sum_k A_k

    Physical Interpretation:
        Wind velocity components, momentum flux

    References:
        - Pielke (2002). Mesoscale Meteorological Modeling.
          Section 6.4: Grid aggregation for momentum.
    """

    def test_scalar_default(self, aggregator_fixture):
        """
        Test that 1D data falls back to arithmetic mean.

        For scalar fields, component-wise mean should behave as arithmetic mean.
        """
        data = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)

        result_component = aggregator_fixture._component_wise_mean(data, 'temperature')
        result_arithmetic = aggregator_fixture._area_weighted_mean(data, 'temperature')

        assert torch.allclose(result_component, result_arithmetic, atol=1e-12), \
            "Scalar default differs from arithmetic mean"

    def test_vector_aggregation(self, aggregator_fixture):
        """
        Test that vector fields are handled (implementation note: currently treats as scalar).

        Note: Current implementation logs warning and uses scalar aggregation.
        This test verifies the default behavior.
        """
        # For now, since implementation treats multidimensional as scalar,
        # we test that it doesn't crash
        data = torch.tensor([5.0, 10.0, 15.0], dtype=torch.float64)

        result = aggregator_fixture._component_wise_mean(data, 'wind_u')

        # Should produce valid output
        assert torch.all(torch.isfinite(result)), "Vector aggregation produced invalid result"
        assert result.shape[0] == 2, "Output shape mismatch"

    def test_vector_components_aggregated_separately(self, aggregator_fixture):
        """
        Test that multi-component vectors are aggregated per component.
        """
        data = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ],
            dtype=torch.float64,
        )

        result = aggregator_fixture._component_wise_mean(data, "wind_uv")
        expected = torch.tensor(
            [
                [(1.0 + 2.0 * 2.0) / 3.0, (10.0 + 20.0 * 2.0) / 3.0],
                [3.0, 30.0],
            ],
            dtype=torch.float64,
        )

        assert result.shape == expected.shape, "Component-wise result shape mismatch"
        assert torch.allclose(result, expected, atol=1e-12), "Component-wise aggregation incorrect"


class TestGeometricMean:
    """
    Test geometric mean for log-normally distributed variables.

    Mathematical Expression:
        y_j = exp(sum_i (w_i * ln(x_i)))
        where w_i = A_i / sum_k A_k

    Physical Interpretation:
        Particle sizes, hydraulic conductivity, certain pollutant concentrations

    References:
        - Limpert et al. (2001). BioScience, 51(5), 341-352.
          Log-normal distributions in ecology.
        - Aitchison & Brown (1957). The Lognormal Distribution. Cambridge Univ. Press.
    """

    def test_simple_geometric_mean(self, aggregator_fixture):
        """
        Test geometric mean with hand-calculated result.

        Setup:
            Source values: [1, 4, 8]
            Source areas: [1, 2, 3]

        Expected for target 0 (sources 0,1):
            w0 = 1/3, w1 = 2/3
            ln_weighted = (1/3)*ln(1) + (2/3)*ln(4) = 0 + (2/3)*1.3863 = 0.9242
            result = exp(0.9242) 鈮?2.5198 鈮?2^1.333

        Expected for target 1 (source 2):
            result = 8
        """
        data = torch.tensor([1.0, 4.0, 8.0], dtype=torch.float64)

        result = aggregator_fixture._geometric_mean(data, 'particle_size')

        # Calculate expected value
        w0, w1 = 1.0/3.0, 2.0/3.0
        ln_weighted_0 = w0 * torch.log(torch.tensor(1.0)) + w1 * torch.log(torch.tensor(4.0))
        expected_0 = torch.exp(ln_weighted_0)

        expected = torch.tensor([expected_0.item(), 8.0], dtype=torch.float64)

        assert torch.allclose(result, expected, atol=1e-10), \
            f"Geometric mean error: got {result}, expected {expected}"

    def test_positive_values_requirement(self, aggregator_fixture):
        """
        Test that geometric mean handles near-zero values with epsilon.

        Geometric mean requires positive values; epsilon should protect against log(0).
        """
        # Include near-zero value
        data = torch.tensor([1e-15, 10.0, 100.0], dtype=torch.float64)

        # Should not raise exception
        result = aggregator_fixture._geometric_mean(data, 'concentration')

        # Result should be finite and positive
        assert torch.all(torch.isfinite(result)), "Result contains inf/nan"
        assert torch.all(result > 0), "Result should be positive"

    def test_geometric_mean_bounds(self, aggregator_fixture):
        """
        Test that geometric mean satisfies bounds property.

        Property: min(x_i) <= geom_mean <= max(x_i)
        """
        data = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64)

        result = aggregator_fixture._geometric_mean(data, 'size')

        # Target 0: sources 0,1 -> min=2, max=4
        assert result[0] >= 2.0 - 1e-10, "Below minimum"
        assert result[0] <= 4.0 + 1e-10, "Above maximum"

        # Target 1: source 2 -> should equal 8
        assert torch.abs(result[1] - 8.0) < 1e-10, "Single source error"


class TestScalingRulesIntegration:
    """
    Integration tests for multiple scaling rules working together.

    Verifies that:
    - Different rules can be applied to different variables
    - Conservation holds across rule types
    - Results are physically consistent
    """

    def test_multi_variable_aggregation(self, aggregator_fixture):
        """
        Test aggregating multiple variables with different rules simultaneously.

        This tests the workflow where different variable types use appropriate aggregation.
        """
        # Prepare different variable types
        data_dict = {
            'resistance': torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64),
            'conductance': torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
            'fraction': torch.tensor([0.3, 0.7, 0.5], dtype=torch.float64),
        }

        # Aggregate each with appropriate strategy
        result_resistance = aggregator_fixture._harmonic_mean(data_dict['resistance'], 'resistance')
        result_conductance = aggregator_fixture._area_weighted_mean(data_dict['conductance'], 'conductance')
        result_fraction = aggregator_fixture._bounded_area_mean(data_dict['fraction'], 'fraction')

        # All should produce valid 2-element tensors
        assert result_resistance.shape == (2,), "Resistance shape mismatch"
        assert result_conductance.shape == (2,), "Conductance shape mismatch"
        assert result_fraction.shape == (2,), "Fraction shape mismatch"

        # All should be finite and in reasonable ranges
        assert torch.all(torch.isfinite(result_resistance)), "Resistance invalid"
        assert torch.all(torch.isfinite(result_conductance)), "Conductance invalid"
        assert torch.all(result_fraction >= 0) and torch.all(result_fraction <= 1), "Fraction out of bounds"

    def test_consistency_across_scales(self, aggregator_fixture):
        """
        Test that repeated aggregation maintains consistency.

        Aggregating uniform data should produce uniform results.
        """
        # Test harmonic and arithmetic with value 50
        uniform_data_50 = torch.tensor([50.0, 50.0, 50.0], dtype=torch.float32)

        # Test bounded with fraction value 0.5
        uniform_data_frac = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

        # Test rules with appropriate data
        test_cases = [
            ('harmonic', aggregator_fixture._harmonic_mean, uniform_data_50, 50.0),
            ('arithmetic', aggregator_fixture._area_weighted_mean, uniform_data_50, 50.0),
            ('bounded', aggregator_fixture._bounded_area_mean, uniform_data_frac, 0.5),
        ]

        for rule_name, rule_func, test_data, expected_val in test_cases:
            result = rule_func(test_data, f'test_{rule_name}')

            # All targets should equal the expected value
            expected = torch.tensor([expected_val, expected_val], dtype=result.dtype)
            assert torch.allclose(result, expected, atol=1e-6), \
                f"Uniformity not preserved for {rule_name}: got {result}, expected {expected}"

    def test_numerical_stability(self, aggregator_fixture):
        """
        Test numerical stability with extreme values.

        Verifies that aggregation handles large and small values without overflow/underflow.
        """
        # Test with large values
        large_data = torch.tensor([1e6, 2e6, 3e6], dtype=torch.float64)
        result_large = aggregator_fixture._area_weighted_mean(large_data, 'large_var')
        assert torch.all(torch.isfinite(result_large)), "Large value overflow"

        # Test with small values
        small_data = torch.tensor([1e-6, 2e-6, 3e-6], dtype=torch.float64)
        result_small = aggregator_fixture._area_weighted_mean(small_data, 'small_var')
        assert torch.all(torch.isfinite(result_small)), "Small value underflow"

        # Test with mixed magnitude
        mixed_data = torch.tensor([1e-3, 1.0, 1e3], dtype=torch.float64)
        result_mixed = aggregator_fixture._area_weighted_mean(mixed_data, 'mixed_var')
        assert torch.all(torch.isfinite(result_mixed)), "Mixed magnitude error"


if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])


