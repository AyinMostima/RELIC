"""
Comprehensive L1 Unit Tests for SurrogateFunctor Base Class

This test suite provides exhaustive testing of the SurrogateFunctor abstract base class,
covering all core functionality, parameter responses, edge cases, and scientific accuracy.
Tests are designed to ensure zero-defect operation and validate the surrogate modeling
framework integration with the RELIC architecture.

Test Categories:
1. Basic Functionality Tests
2. Input/Output Variable Mapping Tests
3. Unit Conversion and CF Standard Name Tests
4. Conservation Law Enforcement Tests
5. Error Handling and Validation Tests
6. Edge Case and Boundary Tests
7. Integration with FunctorBase Tests
8. Configuration and Metadata Tests

Mathematical Framework:
    Surrogate S approximates process F: s -> f
    S: s -> f_hat = S(s, theta) where theta are model parameters
    Conservation constraint: |f_hat| >= 0 for conserved quantities
    Uncertainty: sigma_total = sigma_approx + sigma_param + sigma_input

References:
    - Sacks, J. et al. (1989). Design and Analysis of Computer Experiments
    - O'Hagan, A. (2006). Bayesian analysis of computer code outputs
    - Razavi, S. et al. (2012). Review of surrogate modeling in water resources
    - Raissi, M. et al. (2019). Physics-informed neural networks
"""

import pytest
import torch
import numpy as np
from typing import List, Optional
import sys
import os
from unittest.mock import Mock

# Add relic to Python path for testing
relic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, relic_path)

from relic.models.surrogates import (
    ConservationViolationError,
    ModelLoadError,
    SurrogateError,
    SurrogateFunctor,
)
import relic.models.surrogates as surrogates_module
from relic.core.functor import (
    BiologicalProcessFunctor,
    ConfigDict,
    FunctorBase,
    FunctorMetadata,
    ProcessCategory,
)
from relic.core.variable_mapper import variable_name_mapper


class MockSurrogateFunctor(SurrogateFunctor):
    """
    Concrete implementation of SurrogateFunctor for testing purposes.
    
    This mock implementation provides controllable behavior for testing
    the abstract base class functionality without requiring actual model files.
    """
    
    def __init__(self, 
                 name: str,
                 config: ConfigDict,
                 metadata: Optional[FunctorMetadata] = None,
                 mock_predictions: Optional[torch.Tensor] = None,
                 should_fail_load: bool = False,
                 should_fail_predict: bool = False,
                 **kwargs) -> None:
        """Initialize mock surrogate with controllable behavior."""
        self.mock_predictions = mock_predictions
        self.should_fail_load = should_fail_load
        self.should_fail_predict = should_fail_predict
        self.load_called = False
        self.predict_called = False
        
        super().__init__(name, config, metadata, **kwargs)
    
    def _load_model(self) -> None:
        """Mock model loading with controllable failure."""
        self.load_called = True
        if self.should_fail_load:
            raise ModelLoadError("Mock model loading failed")
        # Simulate successful model loading
        self.model_loaded = True
    
    def _predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Mock prediction with controllable output."""
        self.predict_called = True
        if self.should_fail_predict:
            raise RuntimeError("Mock prediction failed")
        
        if self.mock_predictions is not None:
            # Return mock predictions with correct batch size
            batch_size = input_tensor.shape[0]
            if self.mock_predictions.shape[0] == 1 and batch_size > 1:
                return self.mock_predictions.repeat(batch_size, 1)
            return self.mock_predictions
        
        # Default: return zeros with correct shape
        num_outputs = len(self.config.get('output_variables', ['output']))
        return torch.zeros(input_tensor.shape[0], num_outputs, dtype=input_tensor.dtype)
    
    def get_dependencies(self) -> List[str]:
        """Return mock dependencies."""
        return self.config.get('dependencies', [])


class TestSurrogateFunctorBasicFunctionality:
    """Test basic functionality and interface compliance."""
    
    def test_initialization_default_parameters(self):
        """Test model initialization with minimal configuration."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['air_temperature', 'soil_moisture'],
            'output_variables': ['net_photosynthesis']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # Check basic attributes
        assert surrogate.name == "test_surrogate"
        assert surrogate.config == config
        assert surrogate.load_called  # Model should be loaded during initialization
        assert surrogate.conservation_check is False
        assert surrogate.scale_rate_outputs_by_dt is False
        
        # Check input/output variables are correctly stored
        assert surrogate.input_variables == ['air_temperature', 'soil_moisture']
        assert surrogate.output_variables == ['net_photosynthesis']
    
    def test_initialization_full_configuration(self):
        """Test model initialization with complete configuration."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['air_temperature', 'soil_moisture', 'atmospheric_co2'],
            'output_variables': ['net_photosynthesis', 'stomatal_conductance'],
            'input_mapping': {'air_temperature': 'T_air', 'soil_moisture': 'theta'},
            'output_mapping': {'net_photosynthesis': 'A_net'},
            'units_conversion': {
                'air_temperature': ('celsius', 'kelvin'),
                'net_photosynthesis': ('umol_m2_s1', 'mol_m2_s1')
            },
            'conservation_check': True,
            'conservation_tolerance': 1e-8,
            'conservation_matrix': [[1.0, -1.0]],
            'device': 'cpu',
            'batch_size': 32,
            'uncertainty_quantification': True
        }
        
        metadata = FunctorMetadata(
            name="test_surrogate",
            description="Test surrogate model",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            references=["RELIC surrogate L1 test fixture"],
            input_variables=list(config["input_variables"]),
            output_variables=list(config["output_variables"]),
        )
        
        surrogate = MockSurrogateFunctor("test_surrogate", config, metadata)
        
        # Check all configuration options are stored
        assert surrogate.config['conservation_check'] == True
        assert surrogate.config['conservation_tolerance'] == 1e-8
        assert surrogate.config['device'] == 'cpu'
        assert surrogate.config['batch_size'] == 32
        assert surrogate.config['uncertainty_quantification'] == True
        
        # Check metadata is correctly assigned
        assert surrogate.metadata is metadata
        assert surrogate.metadata.name == "test_surrogate"
        assert surrogate.metadata.description == "Test surrogate model"
        assert surrogate.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS
        assert surrogate.metadata.input_variables == config["input_variables"]
        assert surrogate.metadata.output_variables == [
            "A_net",
            "stomatal_conductance",
        ]
    
    def test_functor_base_inheritance(self):
        """Test that biological surrogates use the canonical lifecycle bridge."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['rate']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # Test FunctorBase interface compliance
        assert isinstance(surrogate, FunctorBase)
        assert isinstance(surrogate, BiologicalProcessFunctor)
        assert hasattr(surrogate, 'forward')
        assert hasattr(surrogate, 'get_dependencies')
        assert hasattr(surrogate, '__call__')
        assert hasattr(surrogate, 'name')
        assert hasattr(surrogate, 'metadata')

    def test_non_biological_metadata_is_rejected(self):
        """A generic surrogate must not bypass a category-specific bridge."""
        config = {
            "model_path": "mock_model.pkl",
            "validate_model_path": False,
            "input_variables": ["temperature"],
            "output_variables": ["rate"],
        }
        metadata = FunctorMetadata(
            name="scale_surrogate",
            description="Invalid scale surrogate on a biological bridge",
            category=ProcessCategory.SCALE_COUPLING,
            references=["RELIC architecture test fixture"],
            input_variables=["temperature"],
            output_variables=["rate"],
        )

        with pytest.raises(TypeError, match="biological_process"):
            MockSurrogateFunctor("scale_surrogate", config, metadata)
    
    def test_abstract_method_enforcement(self):
        """Test that abstract methods must be implemented."""
        from relic.models.surrogates import SurrogateFunctor
        
        config = {
            'model_path': 'test_model.pkl',
            'input_variables': ['input'],
            'output_variables': ['output']
        }
        
        # Should not be able to instantiate abstract class directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SurrogateFunctor("abstract_test", config)


class TestInputOutputMapping:
    """Test input/output variable mapping and preparation."""

    def test_surrogate_module_reuses_canonical_variable_mapper(self):
        """Test surrogate aliases use the ParameterManager-refreshed singleton."""
        assert surrogates_module.variable_mapper is variable_name_mapper

    def test_prepare_input_uses_alias_from_canonical_mapper(self, monkeypatch):
        """Test canonical catalog aliases resolve a surrogate input."""
        monkeypatch.setattr(
            variable_name_mapper,
            "to_standard",
            lambda name: "air_temperature" if name == "tair" else name,
        )
        surrogate = MockSurrogateFunctor(
            "test_surrogate",
            {
                "model_path": "mock_model.pkl",
                "validate_model_path": False,
                "input_variables": ["tair"],
                "output_variables": ["response"],
            },
        )

        prepared = surrogate._prepare_input(
            {"air_temperature": torch.tensor([280.0, 281.0])}
        )

        assert prepared.shape == (2, 1)
        assert torch.equal(
            prepared.cpu(),
            torch.tensor([[280.0], [281.0]], dtype=prepared.dtype),
        )
    
    def test_prepare_input_basic(self):
        """Test basic input preparation from state dictionary."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['air_temperature', 'soil_moisture'],
            'output_variables': ['net_photosynthesis']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # Create test state with required variables
        state = {
            'air_temperature': torch.tensor([20.0, 25.0, 30.0]),
            'soil_moisture': torch.tensor([0.3, 0.4, 0.5]),
            'other_variable': torch.tensor([1.0, 2.0, 3.0])  # Should be ignored
        }
        
        # Test input preparation
        input_tensor = surrogate._prepare_input(state)
        
        # Check tensor shape and content
        assert input_tensor.shape == (3, 2)  # batch_size=3, num_inputs=2
        expected_air = torch.tensor([20.0, 25.0, 30.0], dtype=input_tensor.dtype)
        expected_soil = torch.tensor([0.3, 0.4, 0.5], dtype=input_tensor.dtype)
        assert torch.allclose(input_tensor[:, 0].cpu(), expected_air)
        assert torch.allclose(input_tensor[:, 1].cpu(), expected_soil)
    
    def test_prepare_input_with_mapping(self):
        """Test input preparation with custom variable mapping."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['air_temperature', 'soil_moisture'],
            'output_variables': ['net_photosynthesis'],
            'input_mapping': {
                'air_temperature': 'T_air',
                'soil_moisture': 'theta_soil'
            }
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # State uses standard RELIC names
        state = {
            'air_temperature': torch.tensor([20.0, 25.0]),
            'soil_moisture': torch.tensor([0.3, 0.4])
        }
        
        # Input preparation should work with mapping
        input_tensor = surrogate._prepare_input(state)
        
        assert input_tensor.shape == (2, 2)
        expected_air = torch.tensor([20.0, 25.0], dtype=input_tensor.dtype)
        expected_soil = torch.tensor([0.3, 0.4], dtype=input_tensor.dtype)
        assert torch.allclose(input_tensor[:, 0].cpu(), expected_air)
        assert torch.allclose(input_tensor[:, 1].cpu(), expected_soil)
    
    def test_prepare_input_missing_variable(self):
        """Test error handling when required input variable is missing."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['air_temperature', 'soil_moisture'],
            'output_variables': ['net_photosynthesis']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # State missing required variable
        state = {
            'air_temperature': torch.tensor([20.0, 25.0])
            # Missing 'soil_moisture'
        }
        
        with pytest.raises(KeyError, match="soil_moisture"):
            surrogate._prepare_input(state)

    def test_prepare_input_rejects_implicit_spatial_reduction(self):
        """Spatial aggregation requires an explicit scale-coupling owner."""
        config = {
            "model_path": "mock_model.pkl",
            "validate_model_path": False,
            "input_variables": ["air_temperature"],
            "output_variables": ["net_photosynthesis"],
        }
        surrogate = MockSurrogateFunctor("test_surrogate", config)

        with pytest.raises(ValueError, match="rank-0 or rank-1"):
            surrogate._prepare_input(
                {"air_temperature": torch.ones((2, 3), dtype=torch.float32)}
            )
    
    def test_format_output_basic(self):
        """Test basic output formatting to FluxDict."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['net_photosynthesis', 'respiration']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # Mock model predictions
        predictions = torch.tensor([
            [10.5, 2.3],  # Sample 1: photosynthesis=10.5, respiration=2.3
            [12.1, 2.8]   # Sample 2: photosynthesis=12.1, respiration=2.8
        ])
        
        # Format outputs
        flux_dict = surrogate._format_output(predictions, dt=3600.0)
        
        # Check FluxDict structure
        assert isinstance(flux_dict, dict)
        assert 'net_photosynthesis' in flux_dict
        assert 'respiration' in flux_dict
        
        # Check tensor values
        expected_net = torch.tensor([10.5, 12.1], dtype=flux_dict['net_photosynthesis'].dtype)
        expected_resp = torch.tensor([2.3, 2.8], dtype=flux_dict['respiration'].dtype)
        assert torch.allclose(flux_dict['net_photosynthesis'].cpu(), expected_net)
        assert torch.allclose(flux_dict['respiration'].cpu(), expected_resp)
    
    def test_format_output_with_mapping(self):
        """Test output formatting with custom variable mapping."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['net_photosynthesis'],
            'output_mapping': {'net_photosynthesis': 'A_net'}
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        predictions = torch.tensor([[10.5], [12.1]])
        flux_dict = surrogate._format_output(predictions, dt=3600.0)
        
        # Output should use mapped name
        assert set(flux_dict) == {'A_net'}
        expected_net = torch.tensor([10.5, 12.1], dtype=flux_dict['A_net'].dtype)
        assert torch.allclose(flux_dict['A_net'].cpu(), expected_net)
        assert surrogate.metadata.output_variables == ["A_net"]

    def test_rate_scaling_requires_explicit_output_semantics(self):
        """Rate integration must not be inferred from a variable-name substring."""
        config = {
            "model_path": "mock_model.pkl",
            "validate_model_path": False,
            "input_variables": ["temperature"],
            "output_variables": ["process_rate", "flux_diagnostic"],
            "scale_rate_outputs_by_dt": True,
        }

        with pytest.raises(ValueError, match="rate_output_variables"):
            MockSurrogateFunctor("test_surrogate", config)

        config["rate_output_variables"] = ["process_rate"]
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        outputs = surrogate._format_output(torch.tensor([[2.0, 3.0]]), dt=5.0)

        assert outputs["process_rate"].item() == pytest.approx(10.0)
        assert outputs["flux_diagnostic"].item() == pytest.approx(3.0)

    @pytest.mark.parametrize(
        ("output_variables", "output_mapping", "message"),
        [
            (
                ["net_photosynthesis"],
                {"unknown_output": "A_net"},
                "not in output_variables",
            ),
            (
                ["net_photosynthesis"],
                {"net_photosynthesis": ""},
                "non-empty",
            ),
            (
                ["net_photosynthesis", "respiration"],
                {
                    "net_photosynthesis": "carbon_flux",
                    "respiration": "carbon_flux",
                },
                "unique",
            ),
            (
                ["net_photosynthesis", "A_net"],
                {"net_photosynthesis": "A_net"},
                "unique",
            ),
        ],
    )
    def test_output_mapping_rejects_unknown_empty_or_colliding_targets(
        self,
        output_variables,
        output_mapping,
        message,
    ):
        config = {
            "model_path": "mock_model.pkl",
            "validate_model_path": False,
            "input_variables": ["temperature"],
            "output_variables": output_variables,
            "output_mapping": output_mapping,
        }

        with pytest.raises(ValueError, match=message):
            MockSurrogateFunctor("test_surrogate", config)

    def test_forward_uses_mapped_names_for_values_and_uncertainty(self):
        config = {
            "model_path": "mock_model.pkl",
            "validate_model_path": False,
            "input_variables": ["temperature"],
            "output_variables": ["net_photosynthesis"],
            "output_mapping": {"net_photosynthesis": "A_net"},
            "uncertainty_quantification": True,
            "conservation_check": False,
        }
        surrogate = MockSurrogateFunctor(
            "test_surrogate",
            config,
            mock_predictions=torch.tensor([[10.5]]),
        )
        surrogate._estimate_uncertainty = Mock(
            return_value=torch.tensor([[0.25]])
        )

        outputs = surrogate.forward(
            {"temperature": torch.tensor(280.0)},
            time=0.0,
            dt=3600.0,
        )

        assert set(outputs) == {"A_net", "A_net_uncertainty"}
        assert outputs["A_net"].item() == pytest.approx(10.5)
        assert outputs["A_net_uncertainty"].item() == pytest.approx(0.25)


class TestUnitConversionAndStandards:
    """Test unit conversion and CF standard name handling."""
    
    def test_unit_conversion_temperature(self):
        """Test temperature unit conversion (Celsius to Kelvin)."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['air_temperature'],
            'output_variables': ['rate'],
            'units_conversion': {
                'air_temperature': ('celsius', 'kelvin')
            }
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # State in Celsius
        state = {'air_temperature': torch.tensor([0.0, 25.0, 100.0])}
        
        input_tensor = surrogate._prepare_input(state)
        
        # Should be converted to Kelvin
        expected_kelvin = torch.tensor([273.15, 298.15, 373.15], dtype=input_tensor.dtype)
        assert torch.allclose(input_tensor[:, 0].cpu(), expected_kelvin, atol=1e-6)
    
    def test_unit_conversion_flux_rates(self):
        """Test flux rate unit conversion (umol/m2/s to mol/m2/s)."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['net_photosynthesis'],
            'units_conversion': {
                'net_photosynthesis': ('umol_m2_s1', 'mol_m2_s1')
            }
        }
        
        # Mock predictions in umol/m2/s
        mock_predictions = torch.tensor([[10.0], [20.0]])  # umol/m2/s
        surrogate = MockSurrogateFunctor("test_surrogate", config, 
                                        mock_predictions=mock_predictions)
        
        flux_dict = surrogate._format_output(mock_predictions, dt=3600.0)
        
        # Should be converted to mol/m2/s (divide by 1e6)
        expected_mol = torch.tensor([10e-6, 20e-6], dtype=flux_dict['net_photosynthesis'].dtype)
        assert torch.allclose(flux_dict['net_photosynthesis'].cpu(), expected_mol, atol=1e-9)
    
    @pytest.mark.skipif(not hasattr(sys.modules.get('relic.data.standards.cf_conventions', None), 'CFConventions'),
                       reason="CF conventions module not available")
    def test_cf_standard_name_validation(self):
        """Test CF standard name validation if available."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['air_temperature'],
            'output_variables': ['net_primary_productivity_of_biomass_expressed_as_carbon'],
            'cf_standard_names': True
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # Should not raise error for valid CF standard names
        assert surrogate.config['cf_standard_names'] == True


class TestConservationLaws:
    """Test conservation law enforcement and validation."""
    
    def test_conservation_check_mass_balance(self):
        """Test mass conservation checking for carbon fluxes."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['photosynthesis', 'respiration', 'net_ecosystem_exchange'],
            'conservation_check': True,
            'conservation_tolerance': 1e-6,
            'conservation_matrix': [[1.0, 1.0, -1.0]],
        }
        
        # Mock balanced fluxes: NEE = GPP - Respiration
        mock_predictions = torch.tensor([
            [10.0, -8.0, 2.0],   # GPP=10, Resp=-8, NEE=2 (balanced)
            [15.0, -12.0, 3.0]  # GPP=15, Resp=-12, NEE=3 (balanced)
        ])
        
        surrogate = MockSurrogateFunctor("test_surrogate", config,
                                        mock_predictions=mock_predictions)
        
        state = {'temperature': torch.tensor([20.0, 25.0])}
        
        # Should not raise conservation error
        result = surrogate.forward(state, time=0.0, dt=3600.0)
        
        assert 'photosynthesis' in result
        assert 'respiration' in result
        assert 'net_ecosystem_exchange' in result
    
    def test_conservation_violation_detection(self):
        """Test detection of conservation law violations."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['carbon_flux_in', 'carbon_flux_out'],  # Variables with 'carbon' for detection
            'conservation_check': True,
            'conservation_tolerance': 1e-6,
            'conservation_matrix': [[1.0, 1.0]],
        }
        
        # Mock unbalanced carbon fluxes that violate conservation
        mock_predictions = torch.tensor([
            [10.0, -8.0]   # Net flux = 2.0, large violation
        ])
        
        surrogate = MockSurrogateFunctor("test_surrogate", config,
                                        mock_predictions=mock_predictions)
        
        state = {'temperature': torch.tensor([20.0])}
        
        # Should detect conservation violation
        with pytest.raises(ConservationViolationError, match="2.00e\+00"):
            surrogate.forward(state, time=0.0, dt=3600.0)
    
    def test_conservation_tolerance_handling(self):
        """Test conservation tolerance handling."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['input_flux', 'output_flux'],
            'conservation_check': True,
            'conservation_tolerance': 0.01,
            'conservation_matrix': [[1.0, 1.0]],
        }
        
        # Mock nearly balanced fluxes in a declared, commensurate output ledger.
        mock_predictions = torch.tensor([
            [1.0, -0.999]
        ])
        
        surrogate = MockSurrogateFunctor("test_surrogate", config,
                                        mock_predictions=mock_predictions)
        
        state = {'temperature': torch.tensor([20.0])}
        
        # Should pass with loose tolerance
        result = surrogate.forward(state, time=0.0, dt=3600.0)
        
        assert 'input_flux' in result
        assert 'output_flux' in result

    def test_conservation_check_fails_closed_without_explicit_ledger(self):
        """A strict conservation claim requires a matrix or explicit callback."""
        config = {
            "model_path": "mock_model.pkl",
            "validate_model_path": False,
            "input_variables": ["temperature"],
            "output_variables": ["carbon_flux_in", "carbon_flux_out"],
            "conservation_check": True,
            "conservation_tolerance": 1e-6,
        }
        surrogate = MockSurrogateFunctor(
            "test_surrogate",
            config,
            mock_predictions=torch.tensor([[1.0, -1.0]]),
        )

        with pytest.raises(SurrogateError, match="conservation_matrix"):
            surrogate.forward(
                {"temperature": torch.tensor([20.0])},
                time=0.0,
                dt=1.0,
            )

    @pytest.mark.parametrize(
        "matrix",
        [
            [],
            [[1.0]],
            [[0.0, 0.0]],
            [[float("nan"), 1.0]],
        ],
    )
    def test_conservation_matrix_contract_is_non_vacuous(self, matrix):
        """The declared ledger must be finite, nonzero, and output-aligned."""
        config = {
            "model_path": "mock_model.pkl",
            "validate_model_path": False,
            "input_variables": ["temperature"],
            "output_variables": ["flux_in", "flux_out"],
            "conservation_check": True,
            "conservation_matrix": matrix,
        }

        with pytest.raises(ValueError, match="conservation_matrix"):
            MockSurrogateFunctor("test_surrogate", config)


class TestDeclaredOptionsAreHonoured:
    """A declared uncertainty or conservation option runs or raises."""

    def test_requested_uncertainty_without_an_estimator_is_an_error(self):
        config = {
            "model_path": "mock_model.pkl",
            "validate_model_path": False,
            "input_variables": ["temperature"],
            "output_variables": ["rate"],
            "uncertainty_quantification": True,
        }
        surrogate = MockSurrogateFunctor(
            "test_surrogate", config, mock_predictions=torch.tensor([[1.0]])
        )

        with pytest.raises(NotImplementedError, match="uncertainty"):
            surrogate.forward({"temperature": torch.tensor([20.0])}, time=0.0, dt=1.0)

    def test_conservation_check_requires_an_explicit_tolerance(self):
        config = {
            "model_path": "mock_model.pkl",
            "validate_model_path": False,
            "input_variables": ["temperature"],
            "output_variables": ["flux_in", "flux_out"],
            "conservation_check": True,
            "conservation_matrix": [[1.0, 1.0]],
        }

        with pytest.raises(ValueError, match="conservation_tolerance"):
            MockSurrogateFunctor("test_surrogate", config)


class TestErrorHandlingAndValidation:
    """Test error handling and input validation."""
    
    def test_model_loading_failure(self):
        """Test handling of model loading failures."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['rate']
        }
        
        with pytest.raises(ModelLoadError):
            MockSurrogateFunctor("test_surrogate", config, should_fail_load=True)
    
    @pytest.mark.parametrize(
        "path", ["mock_model.pkl", "latest_model.onnx", "tests/model.onnx"]
    )
    def test_missing_model_file_is_an_error_whatever_its_name(self, path):
        """No path name ('mock', 'test') turns a missing model file into a warning."""
        config = {
            'model_path': path,
            'input_variables': ['temperature'],
            'output_variables': ['rate']
        }

        with pytest.raises(FileNotFoundError, match="model file not found"):
            MockSurrogateFunctor("test_surrogate", config)

    def test_missing_model_file_is_accepted_only_when_declared(self):
        config = {
            'model_path': 'not_yet_trained.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['rate']
        }

        surrogate = MockSurrogateFunctor("test_surrogate", config)

        assert surrogate.load_called

    def test_prediction_failure_handling(self):
        """Test handling of prediction failures during inference."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['rate']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config,
                                        should_fail_predict=True)
        
        state = {'temperature': torch.tensor([20.0])}
        
        with pytest.raises(RuntimeError, match="Mock prediction failed"):
            surrogate.forward(state, time=0.0, dt=3600.0)
    
    def test_invalid_configuration_detection(self):
        """Test detection of invalid configuration parameters."""
        # Missing required configuration keys
        invalid_configs = [
            {},  # Empty config
            {'model_path': 'model.pkl'},  # Missing input_variables
            {'input_variables': ['temp']},  # Missing model_path
            {'model_path': 'model.pkl', 'input_variables': []},  # Empty input_variables
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((KeyError, ValueError, TypeError)):
                MockSurrogateFunctor("test_surrogate", invalid_config)
    
    def test_tensor_type_validation(self):
        """Test validation of tensor types and shapes."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['rate']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # Test with genuinely invalid state types that should fail
        invalid_states = [
            {'temperature': [20.0]},  # List instead of tensor - should fail
            {'temperature': "invalid"},  # String instead of numeric - should fail
            {'temperature': None},  # None instead of numeric - should fail
        ]
        
        for invalid_state in invalid_states:
            with pytest.raises((TypeError, AttributeError, ValueError)):
                surrogate.forward(invalid_state, time=0.0, dt=3600.0)
        
        # Test with valid but non-tensor types that should work
        valid_states = [
            {'temperature': np.array([20.0])},  # NumPy array - should work
            {'temperature': 20.0},  # Scalar - should work
            {'temperature': torch.tensor([20.0])},  # Proper tensor - should work
        ]
        
        for valid_state in valid_states:
            result = surrogate.forward(valid_state, time=0.0, dt=3600.0)
            assert 'rate' in result


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""
    
    def test_empty_state_handling(self):
        """Test handling of empty state dictionaries."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['rate']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        with pytest.raises(KeyError):
            surrogate.forward({}, time=0.0, dt=3600.0)
    
    def test_single_batch_processing(self):
        """Test processing of single sample batches."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['biomass']  # Use non-rate variable to avoid dt scaling
        }
        
        mock_predictions = torch.tensor([[5.0]])
        surrogate = MockSurrogateFunctor("test_surrogate", config,
                                        mock_predictions=mock_predictions)
        
        state = {'temperature': torch.tensor([20.0])}
        result = surrogate.forward(state, time=0.0, dt=3600.0)
        
        assert result['biomass'].shape == torch.Size([1])
        expected_biomass = torch.tensor([5.0], dtype=result['biomass'].dtype)
        assert torch.allclose(result['biomass'].cpu(), expected_biomass)
    
    def test_large_batch_processing(self):
        """Test processing of large batches."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['biomass']  # Use non-rate variable to avoid dt scaling
        }
        
        batch_size = 1000
        mock_predictions = torch.ones(1, 1) * 5.0  # Will be repeated for batch
        surrogate = MockSurrogateFunctor("test_surrogate", config,
                                        mock_predictions=mock_predictions)
        
        state = {'temperature': torch.ones(batch_size) * 20.0}
        result = surrogate.forward(state, time=0.0, dt=3600.0)
        
        assert result['biomass'].shape == torch.Size([batch_size])
        expected_biomass = torch.ones(batch_size, dtype=result['biomass'].dtype) * 5.0
        assert torch.allclose(result['biomass'].cpu(), expected_biomass)
    
    def test_extreme_input_values(self):
        """Test handling of extreme input values."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['rate']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # Test with extreme values
        extreme_states = [
            {'temperature': torch.tensor([float('inf')])},
            {'temperature': torch.tensor([float('-inf')])},
            {'temperature': torch.tensor([float('nan')])},
            {'temperature': torch.tensor([1e10])},
            {'temperature': torch.tensor([-273.15])},  # Absolute zero
        ]
        
        for extreme_state in extreme_states:
            # Should handle gracefully or raise appropriate errors
            try:
                result = surrogate.forward(extreme_state, time=0.0, dt=3600.0)
                # If it succeeds, check that outputs are finite
                assert torch.isfinite(result['rate']).all()
            except (ValueError, RuntimeError):
                # Acceptable to fail on extreme inputs
                pass


class TestIntegrationWithRELICFramework:
    """Test integration with RELIC framework components."""
    
    def test_state_dict_consistency(self):
        """Test consistency with RELIC StateDict format."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['air_temperature', 'soil_water_content'],
            'output_variables': ['gross_primary_productivity']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        # Simulate typical RELIC state
        state = {
            'air_temperature': torch.tensor([15.5, 20.3, 25.1]),
            'soil_water_content': torch.tensor([0.25, 0.30, 0.35]),
            'leaf_area_index': torch.tensor([2.5, 3.0, 3.5]),  # Additional state var
            'photosynthetic_photon_flux': torch.tensor([800, 1000, 1200])
        }
        
        result = surrogate.forward(state, time=3600.0, dt=60.0)
        
        # Check FluxDict format
        assert isinstance(result, dict)
        assert 'gross_primary_productivity' in result
        assert isinstance(result['gross_primary_productivity'], torch.Tensor)
    
    def test_functor_metadata_integration(self):
        """Test integration with FunctorMetadata system."""
        from relic.core.functor import ProcessCategory
        
        metadata = FunctorMetadata(
            name="photosynthesis_surrogate",
            description="ML surrogate for Farquhar photosynthesis model",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            mathematical_expression="A_net = f_ML(T, PAR, CO2, theta)",
            references=["Farquhar et al. (1980)", "Custom ML model (2024)"],
            input_variables=['air_temperature', 'photosynthetic_photon_flux'],
            output_variables=['gross_primary_productivity']
        )
        
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['air_temperature', 'photosynthetic_photon_flux'],
            'output_variables': ['gross_primary_productivity']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config, metadata)
        
        # Check metadata integration
        assert surrogate.metadata.name == "photosynthesis_surrogate"
        assert surrogate.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS
        assert "Farquhar et al. (1980)" in surrogate.metadata.references
        assert surrogate.metadata.mathematical_expression.startswith("Canonical IPHS:")
        assert "equivalence evidence" in surrogate.metadata.mathematical_expression
    
    def test_dependency_declaration(self):
        """Test proper dependency declaration for execution ordering."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['air_temperature', 'vapor_pressure_deficit'],
            'output_variables': ['transpiration'],
            'dependencies': ['radiation_functor', 'microclimate_functor']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        
        dependencies = surrogate.get_dependencies()
        assert 'radiation_functor' in dependencies
        assert 'microclimate_functor' in dependencies
    
    def test_time_and_timestep_handling(self):
        """Test proper handling of time and timestep parameters."""
        config = {
            'model_path': 'mock_model.pkl',
            'validate_model_path': False,
            'input_variables': ['temperature'],
            'output_variables': ['rate']
        }
        
        surrogate = MockSurrogateFunctor("test_surrogate", config)
        state = {'temperature': torch.tensor([20.0])}
        
        # Test various time and dt values
        time_dt_pairs = [
            (0.0, 60.0),      # 1 minute
            (3600.0, 3600.0), # 1 hour
            (86400.0, 86400.0), # 1 day
            (31536000.0, 86400.0) # 1 year time, daily step
        ]
        
        for time, dt in time_dt_pairs:
            result = surrogate.forward(state, time=time, dt=dt)
            assert 'rate' in result
            assert result['rate'].shape == torch.Size([1])


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])


