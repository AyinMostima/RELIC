"""
Comprehensive L1 Unit Tests for Growth Module Parameter System Integration.

Tests verify:
1. All parameters are correctly loaded from ParameterManager
2. Metadata parameters match actual loaded parameters
3. No hardcoded defaults remain
4. Parameter namespaces are correct
5. Conservation tolerances are loaded from validation namespace
6. Default state variable defaults use parameter system
"""

import torch
import yaml
import pytest
from pathlib import Path

# Import test targets
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth
from relic.core.state import UpdateItem

# Import parameter manager
from relic.parameters import get_global_manager


class TestParameterSystemIntegration:
    """Test suite for parameter system integration in growth module."""
    
    def test_biomass_allocation_metadata_parameters(self):
        """Test BiomassAllocation metadata.parameters are descriptive, not runtime values."""
        allocation = BiomassAllocation()
        
        # Check metadata parameters are descriptive (strings or dicts with descriptions)
        assert isinstance(allocation.metadata.parameters, dict)
        for key, value in allocation.metadata.parameters.items():
            if isinstance(value, dict):
                # Parent class may enrich with dict format
                assert 'description' in value, f"Parameter '{key}' dict should have 'description'"
                assert isinstance(value['description'], str), f"Parameter '{key}' description should be a string"
            elif isinstance(value, str):
                # Simple string description is also valid
                pass
            else:
                # Should not be runtime values like bool, float, int
                pytest.fail(f"Parameter '{key}' has invalid type {type(value)}")
    
    def test_validation_conservation_parameters(self):
        """Test conservation tolerance parameters are loaded from validation namespace."""
        allocation = BiomassAllocation()
        param_manager = get_global_manager()
        
        # Create test state with NPP
        state = {
            'net_primary_productivity': torch.tensor(1.0e-6),
            'leaf_area_index': torch.tensor(3.0),
            'phenology_phase': torch.tensor(1.0),
            'air_temperature': torch.tensor(298.15),
            'water_stress_factor': torch.tensor(1.0),
        }
        
        # Run forward pass which should load conservation tolerance
        result = allocation.forward(state, time=0, dt=1)
        
        # Check that validation.conservation parameters were requested
        missing_requests = param_manager.get_missing_requests()
        
        assert isinstance(
            result['diagnostics.biomass_allocation.carbon_balance_residual'],
            UpdateItem,
        )
        assert param_manager.get_parameter(
            'validation.conservation.conservation_tolerance'
        ) is not None
    
    def test_growth_defaults_namespace(self):
        """Test that state ports are not replaced by growth defaults."""
        phenology = PhenologyGrowth(phenology_model_owner='gdd_phenology')
        with pytest.raises(KeyError, match='phenological_factor'):
            phenology.forward({}, time=0, dt=1)

        state = {
            'phenological_factor': torch.tensor(0.6),
        }
        phenology_result = phenology.forward(state, time=0, dt=1)
        assert set(phenology_result) == set(phenology.metadata.output_variables)
        assert 'rate' not in phenology_result
        assert all(isinstance(item, UpdateItem) for item in phenology_result.values())
        
        # StructuralGrowth consumes explicit interconnection state.
        structural = StructuralGrowth()
        result = structural.forward(
            {
                "growth_rate_leaf": torch.tensor(1.0e-7),
                "leaf_area_index": torch.tensor(1.0),
            },
            time=0,
            dt=1,
        )

        assert result
        assert "rate" not in result
        assert all(isinstance(item, UpdateItem) for item in result.values())
    
    def test_phenology_metadata_parameters_loaded(self):
        """Test PhenologyGrowth loads all parameters declared in metadata."""
        phenology = PhenologyGrowth(phenology_model_owner='gdd_phenology')

        process_parameters = {
            key
            for key, entry in phenology.metadata.parameters.items()
            if isinstance(entry, dict)
            and entry.get('namespace', '').startswith('biosphere.growth.')
        }
        assert process_parameters == {'growth_active_threshold'}
        assert isinstance(phenology.growth_active_threshold, float)
        assert 0.0 <= phenology.growth_active_threshold <= 1.0
        for removed in (
            'gdd_budburst', 'gdd_full_leaf', 'gdd_senescence',
            'daylength_critical', 'temperature_optimal',
            'temperature_range_tolerance', 'seconds_per_day',
        ):
            assert not hasattr(phenology, removed)
    
    def test_structural_metadata_parameters_loaded(self):
        """StructuralGrowth declares its exact ParameterManager keys."""
        structural = StructuralGrowth()

        assert 'leaf_turnover_rate' in structural.metadata.parameters
        assert 'senescence_rate' not in structural.metadata.parameters
        assert 'k_senescence' not in structural.metadata.parameters

        manager = get_global_manager()
        for key in (
            'biosphere.phenology.morphology.sla',
            'biosphere.phenology.turnover.leaf_turnover_rate',
        ):
            spec = manager.get_parameter(key)
            assert spec is not None
            assert spec.units
    
    def test_no_hardcoded_environment_defaults(self):
        """Required environmental ports fail fast instead of using defaults."""
        allocation = BiomassAllocation()
        
        # Create state without temperature and water_stress
        state = {
            'net_primary_productivity': torch.tensor(1.0e-6),
            'leaf_area_index': torch.tensor(3.0),
            'phenology_phase': torch.tensor(1.0),
        }

        with pytest.raises(KeyError, match="air_temperature"):
            allocation.forward(state, time=0, dt=1)

        state['air_temperature'] = torch.tensor(298.15)
        with pytest.raises(KeyError, match="water_stress_factor"):
            allocation.forward(state, time=0, dt=1)
        
    def test_parameter_namespace_consistency(self):
        """Biomass metadata separates exact process keys from bridge capabilities."""
        allocation = BiomassAllocation()
        process_parameters = {
            key
            for key, entry in allocation.metadata.parameters.items()
            if isinstance(entry, dict)
            and (
                entry.get('namespace', '').startswith('biosphere.growth.')
                or entry.get('namespace', '').startswith('biosphere.pft.')
                or entry.get('namespace', '').startswith('validation.conservation.')
                or key.startswith('biosphere.pft.')
            )
        }
        assert process_parameters == {
            'f_leaf_base', 'f_stem_base', 'f_root_base',
            'root_alloc_temp_coeff', 'root_alloc_temp_ref',
            'root_alloc_temp_mult', 'water_stress_threshold',
            'temperature_stress_factor', 'temperature_opt',
            'lai_max_threshold', 'lai_min_threshold', 'greenup_f_leaf',
            'dormant_phase_code', 'greenup_phase_code', 'mature_phase_code',
            'senescence_phase_code', 'dormant_phase_modifier',
            'greenup_phase_modifier', 'mature_phase_modifier',
            'senescence_phase_modifier', 'dormant_leaf_allocation',
            'lai_max_leaf_allocation', 'herbaceous_stem_allocation',
            'herbaceous_lai_min_stem_allocation', 'emergency_stem_fraction',
            'emergency_root_fraction', 'default_environmental_modifier',
            'default_phenological_modifier', 'temp_root_factor_min',
            'temp_root_factor_max', 'biosphere.pft.<pft>.allocation.f_leaf',
            'biosphere.pft.<pft>.allocation.f_stem', 'biosphere.pft.<pft>.allocation.f_root',
            'allometry_leaf_stem_exp', 'allometry_root_shoot_exp',
            'allometric_blend_weight', 'min_allocation_frac',
            'max_allocation_frac', 'allocation_sum_tolerance',
            'normalization_epsilon', 'fraction_zero_tolerance',
            'woody_stem_fraction_threshold', 'allometry_biomass_floor',
            'allometry_ratio_max', 'conservation_tolerance',
        }

        # The bridge declares no temperature-law parameters (q10,
        # activation_energy, temperature_sigma): it has no generic temperature
        # law; each process evaluates its own.
        bridge_parameters = set(allocation.metadata.parameters) - process_parameters
        assert bridge_parameters == {
            'ph_optimum',
            'ph_width', 'moisture_half_saturation', 'moisture_linear_min',
            'moisture_linear_max', 'moisture_optimum', 'moisture_width',
        }
    
    def test_parameter_bounds_and_units(self):
        """Test parameters are loaded with correct bounds and units."""
        phenology = PhenologyGrowth(phenology_model_owner='gdd_phenology')
        threshold_spec = get_global_manager().get_parameter(
            'biosphere.growth.allocation.growth_active_threshold'
        )

        assert threshold_spec is not None
        assert threshold_spec.units == 'dimensionless'
        assert threshold_spec.bounds is not None
        assert threshold_spec.bounds[0] <= phenology.growth_active_threshold
        assert phenology.growth_active_threshold <= threshold_spec.bounds[1]
        
        structural = StructuralGrowth()
        result = structural.forward(
            {
                'growth_rate_leaf': torch.tensor(1.0e-7),
                'leaf_area_index': torch.tensor(1.0),
            },
            time=0.0,
            dt=1.0,
        )
        gross = result[
            'diagnostics.structural_growth.gross_leaf_area_growth_tendency'
        ].value
        loss = result[
            'diagnostics.structural_growth.leaf_area_senescence_tendency'
        ].value
        assert gross.item() > 0.0
        assert loss.item() > 0.0
        assert torch.isfinite(gross)
        assert torch.isfinite(loss)
    
    def test_parameter_registration_on_missing(self):
        """Canonical numerical controls are complete ParameterManager entries."""
        param_manager = get_global_manager()
        for key in (
            'normalization_epsilon',
            'fraction_zero_tolerance',
            'woody_stem_fraction_threshold',
            'allometry_biomass_floor',
            'allometry_ratio_max',
        ):
            spec = param_manager.get_parameter(
                f'biosphere.growth.allocation.{key}'
            )
            assert spec is not None
            assert spec.units
            assert spec.bounds is not None
            assert spec.references
            assert spec.provenance


class TestParameterDatabase:
    """Test the parameter database files are correctly structured."""
    
    def test_growth_yaml_exists(self):
        """Test growth parameter YAML files exist and are valid."""
        growth_dir = Path("relic/parameters/database/relic/biosphere/growth")
        assert growth_dir.exists(), "growth directory not found"

        allocation_yaml = growth_dir / "allocation.yaml"
        defaults_yaml = growth_dir / "defaults.yaml"

        for path in (allocation_yaml, defaults_yaml):
            assert path.exists(), f"{path} not found"

        with open(allocation_yaml, 'r', encoding='utf-8') as f:
            allocation = yaml.safe_load(f)
        assert 'parameters' in allocation
        assert 'schema_version' in allocation

        # Representative core growth allocation parameters
        for param in ['f_leaf_base', 'lai_max_threshold', 'height_diameter_coeff_a']:
            assert param in allocation['parameters'], f"Parameter {param} not in allocation.yaml"

        with open(defaults_yaml, 'r', encoding='utf-8') as f:
            defaults = yaml.safe_load(f)
        assert 'parameters' in defaults
        assert 'schema_version' in defaults
        for removed in [
            'default_growth_rate_leaf',
            'default_current_lai',
            'lai_change_rate_default',
            'leaf_area_growth_default',
        ]:
            assert removed not in defaults['parameters']

    def test_growth_rate_diagnostic_defaults_are_retired(self):
        """BiomassAllocation computes its organ and total growth rates [kg C m-2 s-1]
        from NPP on every call and no code reads a default for them, so the growth
        database declares no growth-rate diagnostic defaults."""
        growth_dir = Path("relic/parameters/database/relic/biosphere/growth")
        assert not (growth_dir / "diagnostics.yaml").exists()

        retired_namespace = "biosphere.growth.diagnostics.allocation."
        declared = []
        for path in Path("relic/parameters/database/relic/biosphere").rglob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            namespace = str(payload.get("namespace") or "")
            for key in (payload.get("parameters") or {}):
                full_key = str(key)
                if namespace and not full_key.startswith(namespace + "."):
                    full_key = f"{namespace}.{full_key}"
                if full_key.startswith(retired_namespace):
                    declared.append(full_key)
        assert declared == []
    
    def test_validation_conservation_yaml_exists(self):
        """Test validation/conservation.yaml exists and is valid."""
        yaml_path = Path("relic/parameters/database/relic/validation/conservation.yaml")
        assert yaml_path.exists(), "validation/conservation.yaml not found"
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        assert 'parameters' in data
        
        # Check carbon conservation tolerance exists
        carbon_tolerance = 'validation.conservation.carbon_conservation_tolerance'
        assert carbon_tolerance in data['parameters']
        param_data = data['parameters'][carbon_tolerance]
        # Value might be stored as string or float
        if isinstance(param_data['value'], str):
            assert float(param_data['value']) == 1e-6
        else:
            assert param_data['value'] == 1e-6
        assert param_data['units'] == 'dimensionless'
    
    def test_growth_defaults_in_yaml(self):
        """Test default parameters are in growth defaults.yaml."""
        defaults_path = Path("relic/parameters/database/relic/biosphere/growth/defaults.yaml")

        with open(defaults_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        assert 'parameters' in data

        expected_defaults = [
            'temperature_optimal',
            'temperature_range_tolerance',
        ]

        for param in expected_defaults:
            assert param in data['parameters'], f"Default parameter {param} not in defaults.yaml"
            param_data = data['parameters'][param]
            assert 'value' in param_data
            assert 'units' in param_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

