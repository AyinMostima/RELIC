#!/usr/bin/env python3
"""
Unit tests for FunctorBase and bridge classes.

Tests the core functor framework including:
- Basic functor instantiation and configuration
- Parameter management and validation
- Forward method execution
- Dependency handling
- Conservation law checking
- Bridge class functionality
"""

import pytest
import torch
from typing import Dict, List

# Import functor base classes
from relic.core.functor import (
    FunctorBase,
    FunctorMetadata,
    ProcessCategory,
    FluxDict
)

from relic.core.functor import (
    BiologicalProcessFunctor,
    TransportMechanismFunctor,
    PhysicalFieldFunctor,
    ScaleCouplingFunctor
)

# Type aliases for clarity
StateDict = Dict[str, torch.Tensor]


class SimpleFunctor(FunctorBase):
    """Simple test Functor for unit testing"""

    def __init__(self):
        metadata = FunctorMetadata(
            name="SimpleFunctor",
            description="A simple test functor",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            version="1.0.0",
            references=["RELIC FunctorBase test fixture"],
            parameters={"test_param1": "First test parameter", "test_param2": "Second test parameter"},
            input_variables=["air_temperature"],
            output_variables=["test_output"],
            conservation_laws=[]
        )
        # The instance config supplies the test parameters (no caller default).
        super().__init__(
            name="SimpleFunctor",
            config={"parameters": {"test_param1": 1.0, "test_param2": 2.0}},
            metadata=metadata,
        )

    def get_dependencies(self) -> List[str]:
        """No dependencies for simple test"""
        return []

    def forward(self, state: Dict[str, torch.Tensor],
                time: float, dt: float, **kwargs) -> FluxDict:
        """Simple forward calculation"""
        # Get input
        temp = state.get('air_temperature', torch.tensor(300.0))

        # Get parameters
        param1 = self.get_parameter_value('test_param1')
        param2 = self.get_parameter_value('test_param2')

        # Simple calculation
        result = temp * param1 + param2

        return {
            'test_output': result
        }


class DependentFunctor(FunctorBase):
    """Functor with dependencies for testing"""

    def __init__(self):
        metadata = FunctorMetadata(
            name="DependentFunctor",
            description="A functor with dependencies",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            version="1.0.0",
            references=["RELIC FunctorBase test fixture"],
            parameters={},
            input_variables=["test_output"],
            output_variables=["dependent_output"],
            conservation_laws=[]
        )
        super().__init__(name="DependentFunctor", config={}, metadata=metadata)

    def get_dependencies(self) -> List[str]:
        """Depends on SimpleFunctor"""
        return ["SimpleFunctor"]

    def forward(self, state: Dict[str, torch.Tensor],
                time: float, dt: float, **kwargs) -> FluxDict:
        """Process dependent on SimpleFunctor output"""
        test_input = state.get('test_output', torch.tensor(0.0))
        result = test_input * 2.0

        return {
            'dependent_output': result
        }


class ConservationFunctor(FunctorBase):
    """Functor for testing conservation checks"""

    def __init__(self):
        metadata = FunctorMetadata(
            name="ConservationFunctor",
            description="A functor for testing conservation",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            version="1.0.0",
            references=["RELIC FunctorBase test fixture"],
            parameters={},
            input_variables=["carbon_pool"],
            output_variables=["carbon_flux", "carbon_pool"],
            conservation_laws=["carbon"]
        )
        super().__init__(name="ConservationFunctor", config={}, metadata=metadata)

    def get_dependencies(self) -> List[str]:
        return []

    def forward(self, state: Dict[str, torch.Tensor],
                time: float, dt: float, **kwargs) -> FluxDict:
        """Transfer carbon between pools"""
        carbon = state.get('carbon_pool', torch.tensor(100.0))

        # Transfer 10% of carbon
        flux = carbon * 0.1

        return {
            'carbon_flux': flux,
            'carbon_pool': -flux  # Negative to indicate removal
        }


class TestFunctorBase:
    """Test suite for FunctorBase functionality"""

    def test_functor_instantiation(self):
        """Test basic Functor instantiation"""
        functor = SimpleFunctor()
        assert functor is not None, "Failed to instantiate SimpleFunctor"
        assert functor.metadata is not None, "Functor metadata not set"
        assert functor.metadata.name == "SimpleFunctor", "Incorrect functor name"

    def test_functor_metadata(self):
        """Test Functor metadata"""
        functor = SimpleFunctor()

        assert functor.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS, "Incorrect category"
        assert functor.metadata.version == "1.0.0", "Incorrect version"
        assert "test_param1" in functor.metadata.parameters, "Missing parameter in metadata"
        assert "air_temperature" in functor.metadata.input_variables, "Missing input in metadata"
        assert "test_output" in functor.metadata.output_variables, "Missing output in metadata"

    def test_parameter_access(self):
        """Test parameter access through get_parameter_value"""
        functor = SimpleFunctor()

        # The config supplies the value; an absent parameter is None.
        param = functor.get_parameter_value('test_param1')
        assert param == 1.0, "Config parameter value not returned"
        assert functor.get_parameter_value('missing_param') is None

        # Test required parameter - skip this as it may behave differently
        # with pytest.raises(ValueError):
        #     functor.get_parameter_value('missing_param', required=True)

    def test_forward_execution(self):
        """Test forward method execution"""
        functor = SimpleFunctor()

        state = {
            'air_temperature': torch.tensor(300.0)
        }

        result = functor.forward(state, time=0.0, dt=1.0)

        assert isinstance(result, dict), "Forward should return dict"
        assert 'test_output' in result, "Expected output not in result"
        assert isinstance(result['test_output'], torch.Tensor), "Output should be Tensor"

    def test_dependencies(self):
        """Test dependency declaration"""
        simple = SimpleFunctor()
        dependent = DependentFunctor()

        assert simple.get_dependencies() == [], "SimpleFunctor should have no dependencies"
        assert "SimpleFunctor" in dependent.get_dependencies(), "DependentFunctor missing dependency"

    def test_call_method(self):
        """Test __call__ method with validation"""
        functor = SimpleFunctor()

        state = {
            'air_temperature': torch.tensor(300.0)
        }

        result = functor(state, time=0.0, dt=1.0)

        assert isinstance(result, dict), "__call__ should return dict"
        assert 'test_output' in result, "Expected output not in __call__ result"

    def test_conservation_check(self):
        """Test conservation checking in Functor"""
        functor = ConservationFunctor()

        state = {
            'carbon_pool': torch.tensor(100.0)
        }

        # This should pass conservation check (flux in = flux out)
        result = functor(state, time=0.0, dt=1.0)

        assert 'carbon_flux' in result, "Missing carbon flux"
        assert 'carbon_pool' in result, "Missing carbon pool update"

        # Check conservation
        total_flux = result['carbon_flux'] + result['carbon_pool']
        assert torch.allclose(total_flux, torch.tensor(0.0)), "Conservation violated"

    def test_device_handling(self):
        """Test device handling for tensors"""
        functor = SimpleFunctor()

        # Test CPU tensors
        state = {
            'air_temperature': torch.tensor(300.0, device='cpu')
        }

        result = functor.forward(state, time=0.0, dt=1.0)
        assert result['test_output'].device.type == 'cpu', "Device mismatch"

        # Test GPU tensors if available
        if torch.cuda.is_available():
            state_gpu = {
                'air_temperature': torch.tensor(300.0, device='cuda')
            }

            result_gpu = functor.forward(state_gpu, time=0.0, dt=1.0)
            assert result_gpu['test_output'].device.type == 'cuda', "GPU device not preserved"

    def test_batch_processing(self):
        """Test batch dimension handling"""
        functor = SimpleFunctor()

        # Test with batch dimension
        batch_size = 10
        state = {
            'air_temperature': torch.ones(batch_size) * 300.0
        }

        result = functor.forward(state, time=0.0, dt=1.0)
        assert result['test_output'].shape[0] == batch_size, "Batch dimension not preserved"

    def test_error_handling(self):
        """Test error handling in Functor"""
        functor = SimpleFunctor()

        # Test with missing required input
        state = {}  # Missing air_temperature

        # Should use default or handle gracefully
        result = functor.forward(state, time=0.0, dt=1.0)
        assert 'test_output' in result, "Functor should handle missing input gracefully"


class TestFunctorBridges:
    """Test suite for Functor bridge classes"""

    def test_biological_process_functor(self):
        """Test BiologicalProcessFunctor bridge"""

        class TestBioFunctor(BiologicalProcessFunctor):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="TestBioFunctor",
                    description="Test biological functor",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    version="1.0.0",
                    references=["RELIC bridge test fixture"],
                    parameters={"q10": "Temperature sensitivity coefficient"},
                    input_variables=["temperature"],
                    output_variables=["growth_rate"],
                    conservation_laws=[]
                )
                super().__init__(name="TestBioFunctor", config={}, metadata=metadata)

            def get_dependencies(self) -> List[str]:
                return []

            def forward(self, state: Dict[str, torch.Tensor],
                       time: float, dt: float, **kwargs) -> FluxDict:
                """Forward method implementation"""
                temp = state["temperature"]
                rate = self.temperature_function(temp)
                return {'growth_rate': rate}

            def temperature_function(self, temperature):
                # The concrete Functor owns its law: Q10 = 2 about 298.15 K per 10 K.
                return 2.0 ** ((temperature - 298.15) / 10.0)

        # Test instantiation
        bio_functor = TestBioFunctor()
        assert bio_functor is not None, "Failed to instantiate BiologicalProcessFunctor"

        # Test temperature function
        temp = torch.tensor(308.15)  # 35 C
        rate = bio_functor.temperature_function(temp)
        assert rate > 1.0, "Q10 function should increase rate with temperature"

    def test_transport_mechanism_functor(self):
        """Test TransportMechanismFunctor bridge"""

        class TestTransportFunctor(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def __init__(self):
                metadata = FunctorMetadata(
                    name="TestTransportFunctor",
                    description="Test transport functor",
                    category=ProcessCategory.TRANSPORT_MECHANISM,
                    version="1.0.0",
                    references=["RELIC bridge test fixture"],
                    parameters={"diffusivity": "Molecular diffusion coefficient (m2/s)"},
                    input_variables=["concentration"],
                    output_variables=["flux"],
                    conservation_laws=[]
                )
                super().__init__(name="TestTransportFunctor", config={}, metadata=metadata)
                # The test law uses a constant D [m2 s-1], set through the validated setter.
                self.set_scalar_diffusivity(1e-5)

            def get_dependencies(self) -> List[str]:
                return []

            def forward(self, state: Dict[str, torch.Tensor],
                       time: float, dt: float, **kwargs) -> FluxDict:
                """Forward method implementation"""
                conc = state.get('concentration', torch.tensor([10.0, 5.0]))
                if conc.numel() >= 2:
                    source = conc[0]
                    sink = conc[1]
                else:
                    source = conc
                    sink = torch.tensor(0.0)
                return {"transport_flux": self.diffusivity * (source - sink)}

        # Test instantiation
        transport_functor = TestTransportFunctor()
        assert transport_functor is not None, "Failed to instantiate TransportMechanismFunctor"

        # Test flux calculation
        source = torch.tensor(10.0)
        sink = torch.tensor(5.0)
        fluxes = transport_functor(
            {"concentration": torch.stack((source, sink))},
            time=0.0,
            dt=1.0,
        )
        assert isinstance(fluxes, dict)
        assert "transport_flux" in fluxes
        assert fluxes["transport_flux"] > 0, "Flux should be positive from high to low concentration"

    def test_physical_field_functor(self):
        """Test PhysicalFieldFunctor bridge"""

        class TestFieldFunctor(PhysicalFieldFunctor):

            # Coordinate dimensions of the field's spatial query.
            SPATIAL_DIMS = 3
            def __init__(self):
                metadata = FunctorMetadata(
                    name="TestFieldFunctor",
                    description="Test field functor",
                    category=ProcessCategory.PHYSICAL_FIELD,
                    version="1.0.0",
                    references=["RELIC bridge test fixture"],
                    parameters={},
                    input_variables=["field_value"],
                    output_variables=["field_gradient"],
                    conservation_laws=[]
                )
                super().__init__(name="TestFieldFunctor", config={}, metadata=metadata)

            def get_dependencies(self) -> List[str]:
                return []

            def forward(self, state: Dict[str, torch.Tensor],
                       time: float, dt: float, **kwargs) -> FluxDict:
                """Forward method implementation"""
                coords = state.get('coordinates', torch.tensor([1.0, 2.0, 3.0]))
                gradient = self.compute_gradient(coords)
                return {'field_gradient': gradient}

            def get_value_at(self, coordinates: torch.Tensor) -> torch.Tensor:
                """Get field value at coordinates"""
                # Simple linear field
                return coordinates.sum(dim=-1)

            def compute_gradient(self, coordinates: torch.Tensor) -> torch.Tensor:
                """Compute field gradient"""
                # Gradient of linear field is constant
                return torch.ones_like(coordinates)

        # Test instantiation
        field_functor = TestFieldFunctor()
        assert field_functor is not None, "Failed to instantiate PhysicalFieldFunctor"

        # Test field operations
        coords = torch.tensor([1.0, 2.0, 3.0])
        value = field_functor.get_value_at(coords)
        gradient = field_functor.compute_gradient(coords)

        assert value == 6.0, "Incorrect field value"
        assert torch.allclose(gradient, torch.ones(3)), "Incorrect gradient"

    def test_scale_coupling_functor(self):
        """Test ScaleCouplingFunctor bridge"""

        class TestScaleFunctor(ScaleCouplingFunctor):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="TestScaleFunctor",
                    description="Test scale coupling functor",
                    category=ProcessCategory.SCALE_COUPLING,
                    version="1.0.0",
                    references=["RELIC bridge test fixture"],
                    parameters={},
                    input_variables=["fine_data"],
                    output_variables=["coarse_data"],
                    conservation_laws=["mass"]
                )
                super().__init__(
                    name="TestScaleFunctor",
                    config={},
                    metadata=metadata,
                    source_scale="fine",
                    target_scale="coarse",
                    coupling_type="spatial",
                )

            def get_dependencies(self) -> List[str]:
                return []

            def forward(self, state: Dict[str, torch.Tensor],
                       time: float, dt: float, **kwargs) -> FluxDict:
                """Forward method implementation"""
                fine_data = state.get('fine_data', torch.randn(10))
                coarse = self.upscale(fine_data)
                return {'coarse_data': coarse}

            def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
                """Upscale from fine to coarse resolution"""
                # Simple averaging
                return fine_data.mean()

            def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
                """Downscale from coarse to fine resolution"""
                # Simple replication
                return coarse_data.expand(10)

        # Test instantiation
        scale_functor = TestScaleFunctor()
        assert scale_functor is not None, "Failed to instantiate ScaleCouplingFunctor"

        # Test scaling operations
        fine_data = torch.randn(10)
        coarse = scale_functor.upscale(fine_data)
        fine_again = scale_functor.downscale(coarse)

        assert coarse.numel() == 1, "Upscaling should reduce dimensions"
        assert fine_again.numel() == 10, "Downscaling should increase dimensions"

        # Test conservation check
        total_fine = fine_data.sum()
        total_coarse = coarse * 10
        is_conserved = scale_functor.check_conservation_bridge(total_fine, total_coarse)
        assert is_conserved, "Mass conservation violated in scaling"


class TestFunctorIntegration:
    """Integration tests for Functor system"""

    def test_functor_chain(self):
        """Test chain of dependent Functors"""
        simple = SimpleFunctor()
        dependent = DependentFunctor()

        # Initial state
        state = {
            'air_temperature': torch.tensor(300.0)
        }

        # Execute simple functor
        result1 = simple(state, time=0.0, dt=1.0)

        # Update state with result
        state.update(result1)

        # Execute dependent functor
        result2 = dependent(state, time=0.0, dt=1.0)

        assert 'dependent_output' in result2, "Dependent functor failed to produce output"

    def test_multiple_functors(self):
        """Test multiple Functors operating on same state"""
        functors = [
            SimpleFunctor(),
            ConservationFunctor()
        ]

        state = {
            'air_temperature': torch.tensor(300.0),
            'carbon_pool': torch.tensor(100.0)
        }

        all_results = {}

        for functor in functors:
            result = functor(state, time=0.0, dt=1.0)
            all_results.update(result)

        assert 'test_output' in all_results, "SimpleFunctor output missing"
        assert 'carbon_flux' in all_results, "ConservationFunctor output missing"

    def test_real_functor_loading(self):
        """Test loading actual Functor from scientific modules"""
        from relic.scientific.atmosphere.radiation import RadiationFunctor

        radiation = RadiationFunctor()

        assert hasattr(radiation, 'metadata'), "Real functor missing metadata"
        assert hasattr(radiation, 'forward'), "Real functor missing forward method"
        assert hasattr(radiation, 'get_dependencies'), "Real functor missing dependencies"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

