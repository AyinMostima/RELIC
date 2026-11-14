"""
L1 Unit Tests for Bridge Base Classes

This module contains unit tests for the newly implemented bridge base classes
that unify FunctorBase execution shell with the four abstract base classes.

Tests focus on:
1. Bridge class instantiation and initialization
2. Method resolution order (MRO) correctness
3. Parameter access through unified interface
4. Abstract base class method availability
5. Automatic metadata and category assignment
6. Unit conversion utilities
7. Device consistency and tensor handling

Test Structure:
- TestBiologicalProcessFunctor: Tests for biological process bridge
- TestTransportMechanismFunctor: Tests for transport mechanism bridge  
- TestPhysicalFieldFunctor: Tests for physical field bridge
- TestScaleCouplingFunctor: Tests for scale coupling bridge
- TestUnitConverter: Tests for CF-1.8 unit conversion
- TestBridgeIntegration: Integration tests between bridges

References:
    - pytest documentation for test structure
    - PyTorch testing utilities for tensor assertions
    - Python unittest.mock for dependency mocking
"""

import pytest
import torch

from relic.core.functor import (
    BiologicalProcessFunctor,
    TransportMechanismFunctor,
    PhysicalFieldFunctor,
    ScaleCouplingFunctor,
    FunctorMetadata,
    ProcessCategory,
)
from relic.data.standards.unit_converter import UnitConverter
from relic.utils.physical_constants import get_physical_constant

def _metadata(
    name: str,
    category: ProcessCategory,
    *,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
) -> FunctorMetadata:
    """Create complete metadata for bridge test fixtures."""
    return FunctorMetadata(
        name=name,
        description=f"Bridge L1 fixture for {name}.",
        category=category,
        references=["RELIC architecture bridge contract test."],
        input_variables=list(inputs),
        output_variables=list(outputs),
    )


class TestBiologicalProcessFunctor:
    """Test biological process bridge class functionality."""
    
    def test_initialization_basic(self):
        """Test basic initialization of biological process functor."""
        config = {'parameters': {'vcmax25': 60.0, 'jmax25': 120.0}}
        
        class TestPhotosynthesis(BiologicalProcessFunctor):
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                return {'gpp': torch.tensor(1.0)}
        
        functor = TestPhotosynthesis(
            name='test_photosynthesis',
            config=config,
            metadata=_metadata(
                'test_photosynthesis',
                ProcessCategory.BIOLOGICAL_PROCESS,
                outputs=('gpp',),
            ),
        )
        
        # Test basic properties
        assert functor.name == 'test_photosynthesis'
        assert functor.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS
        assert hasattr(functor, 'process_name')
        assert hasattr(functor, 'q10_value')
    
    def test_metadata_category_preservation(self):
        """Test explicit metadata category preservation."""
        class TestProcess(BiologicalProcessFunctor):
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                return {'rate': torch.tensor(1.0)}
        
        functor1 = TestProcess(
            name='test1',
            config={},
            metadata=_metadata(
                'test1',
                ProcessCategory.BIOLOGICAL_PROCESS,
                outputs=('rate',),
            ),
        )
        assert functor1.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS
        
        metadata = _metadata(
            'test2',
            ProcessCategory.BIOLOGICAL_PROCESS,
            outputs=('rate',),
        )
        functor2 = TestProcess(name='test2', config={}, metadata=metadata)
        assert functor2.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS
    
    def test_parameter_access(self):
        """Test unified parameter access through bridge class."""
        config = {
            'parameters': {
                'vcmax25': 75.0,
                'test_param': {'value': 42.0}
            }
        }
        
        class TestProcess(BiologicalProcessFunctor):
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                return {'rate': torch.tensor(1.0)}
        
        functor = TestProcess(
            name='test',
            config=config,
            metadata=_metadata(
                'test',
                ProcessCategory.BIOLOGICAL_PROCESS,
                outputs=('rate',),
            ),
        )
        
        vcmax = functor.get_parameter_value(
            'vcmax25',
            source_order=('config',),
            required=True,
        )
        assert vcmax == 75.0

        test_param = functor.get_parameter_value(
            'test_param',
            source_order=('config',),
            required=True,
        )
        assert test_param == 42.0

        with pytest.raises(ValueError, match="Required parameter"):
            functor.get_parameter_value(
                'missing_param',
                source_order=('config',),
                required=True,
            )
    
    def test_forward_single_entry_contract(self):
        """Bridge API should expose forward as the only compute entry."""
        class TestProcess(BiologicalProcessFunctor):
            def get_dependencies(self):
                return []

            def temperature_function(self, temp):
                # The concrete Functor owns its law: Q10 = 2 about 298.15 K per 10 K.
                return 2.0 ** ((temp - 298.15) / 10.0)

            def forward(self, state, time, dt, **kwargs):
                temp = state["temperature"]
                substrate = state["substrate"]
                return {"substrate_rate": substrate * self.temperature_function(temp)}
        
        functor = TestProcess(
            name='test',
            config={},
            metadata=_metadata(
                'test',
                ProcessCategory.BIOLOGICAL_PROCESS,
                inputs=('substrate', 'temperature'),
                outputs=('substrate_rate',),
            ),
        )
        
        state = {'substrate': torch.tensor(10.0)}
        environment = {'temperature': torch.tensor(298.15)}

        merged_state = dict(state)
        merged_state.update(environment)
        result = functor.forward(merged_state, time=0.0, dt=1.0)

        assert isinstance(result, dict)
        assert "substrate_rate" in result
        assert not hasattr(functor, "compute_rate_bridge")


class TestTransportMechanismFunctor:
    """Test transport mechanism bridge class functionality."""
    
    def test_initialization_basic(self):
        """Test basic initialization of transport mechanism functor."""
        config = {}
        
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "gas_exchange"

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {'flux': torch.tensor(0.1)}

        functor = TestTransport(
            name='test_transport',
            config=config,
            peclet_number=10.0,
            metadata=_metadata(
                'test_transport',
                ProcessCategory.TRANSPORT_MECHANISM,
                outputs=('flux',),
            ),
        )
        
        # Test basic properties
        assert functor.name == 'test_transport'
        assert functor.metadata.category == ProcessCategory.TRANSPORT_MECHANISM
        assert functor.transport_type == 'gas_exchange'
        assert functor.diffusivity is None
        functor.set_scalar_diffusivity(2e-5)
        assert functor.diffusivity == pytest.approx(2e-5)
        assert functor.peclet_number == pytest.approx(10.0)
    
    def test_flux_to_updates(self):
        """Test flux to updates conversion."""
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                return {'flux': torch.tensor(0.1)}
        
        functor = TestTransport(
            name='test',
            config={},
            metadata=_metadata(
                'test',
                ProcessCategory.TRANSPORT_MECHANISM,
                outputs=('flux',),
            ),
        )
        
        fluxes = {'diffusive_flux': torch.tensor([1.0, 2.0])}
        updates = functor.flux_to_updates(fluxes)
        
        # Default implementation should be identity
        assert updates == fluxes
    
    def test_transport_validation(self, caplog):
        """Test transport mechanism validation methods."""
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                return {'flux': torch.tensor(0.1)}
        
        functor = TestTransport(
            name='test',
            config={},
            metadata=_metadata(
                'test',
                ProcessCategory.TRANSPORT_MECHANISM,
                outputs=('flux',),
            ),
        )
        
        # Test validation with tensor flux
        fluxes = {'transport_flux': torch.tensor([1.0, 2.0])}
        
        assert functor.validate_transport_units(fluxes) is None
        
        invalid_fluxes = {'invalid_flux': [1.0, 2.0]}
        with caplog.at_level(
            'WARNING',
            logger='relic.core.functor.transport_bridge',
        ):
            assert functor.validate_transport_units(invalid_fluxes) is None
        assert 'Non-tensor flux in transport test: invalid_flux' in caplog.text


class TestPhysicalFieldFunctor:
    """Test physical field bridge class functionality."""
    
    def test_initialization_basic(self):
        """Test basic initialization of physical field functor."""
        config = {}
        
        class TestField(PhysicalFieldFunctor):
            SPATIAL_DIMS = 3

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {'field_update': torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

            def get_value_at(self, coordinates):
                return coordinates[..., 0]

        functor = TestField(
            name='test_field',
            config=config,
            metadata=_metadata(
                'test_field',
                ProcessCategory.PHYSICAL_FIELD,
                outputs=('field_update',),
            ),
        )

        # Test basic properties: the field identity is the declared output and
        # SPATIAL_DIMS is the class attribute.
        assert functor.name == 'test_field'
        assert functor.metadata.category == ProcessCategory.PHYSICAL_FIELD
        assert functor.metadata.output_variables == ['field_update']
        assert functor.SPATIAL_DIMS == 3
        assert not hasattr(functor, 'field_name')
        assert not hasattr(functor, 'spatial_dims')
    
    def test_update_field_to_flux(self):
        """Test field update to flux conversion."""
        class TestField(PhysicalFieldFunctor):
            SPATIAL_DIMS = None

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {'field_update': torch.tensor([1.0])}

            def get_value_at(self, coordinates):
                raise NotImplementedError("point model (SPATIAL_DIMS = None): no coordinate query")

        functor = TestField(
            name='temperature_field',
            config={},
            metadata=_metadata(
                'temperature_field',
                ProcessCategory.PHYSICAL_FIELD,
                outputs=('field_update',),
            ),
        )

        field_update = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        flux_dict = functor.update_field_to_flux(field_update, 4.0, 'field_update')

        assert isinstance(flux_dict, dict)
        assert set(flux_dict) == {'field_update'}
        assert torch.equal(flux_dict['field_update'], field_update / 4.0)
        # The single-argument form (fabricated '<field_name>_update' key) is gone.
        with pytest.raises(TypeError):
            functor.update_field_to_flux(field_update)


class TestScaleCouplingFunctor:
    """Test scale coupling bridge class functionality."""
    
    def test_initialization_basic(self):
        """Test basic initialization of scale coupling functor."""
        config = {}
        
        class TestCoupling(ScaleCouplingFunctor):
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                return {'upscaled_data': torch.tensor([1.0, 2.0])}
        
        functor = TestCoupling(
            name='test_coupling',
            config=config,
            source_scale='plot',
            target_scale='grid',
            coupling_type='hierarchical',
            metadata=_metadata(
                'test_coupling',
                ProcessCategory.SCALE_COUPLING,
                outputs=('upscaled_data',),
            ),
        )
        
        # Test basic properties
        assert functor.name == 'test_coupling'
        assert functor.metadata.category == ProcessCategory.SCALE_COUPLING
        assert functor.source_scale == 'plot'
        assert functor.target_scale == 'grid'
        assert functor.coupling_type == 'hierarchical'
    
    def test_conservation_bridge(self):
        """Test conservation checking bridge method."""
        class TestCoupling(ScaleCouplingFunctor):
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                return {'data': torch.tensor([1.0])}
        
        functor = TestCoupling(
            name='test',
            config={},
            metadata=_metadata(
                'test',
                ProcessCategory.SCALE_COUPLING,
                outputs=('data',),
            ),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        )
        
        # Test exact conservation
        input_data = torch.tensor([1.0, 2.0, 3.0])
        output_data = torch.tensor([6.0])  # Sum preserved
        
        is_conserved = functor.check_conservation_bridge(input_data, output_data)
        assert is_conserved  # Should pass conservation check
        
        # Test conservation violation
        output_data_bad = torch.tensor([10.0])  # Sum not preserved
        is_conserved_bad = functor.check_conservation_bridge(input_data, output_data_bad)
        assert not is_conserved_bad  # Should fail conservation check


class TestUnitConverter:
    """Test unit conversion utilities."""
    
    def test_soil_moisture_conversion(self):
        """Test soil moisture unit conversions between m3/m3 and kg/m2."""
        
        # Test m3/m3 to kg/m2 conversion
        volume_fraction = torch.tensor([0.3, 0.4])  # 30%, 40%
        soil_depth = 1.0  # 1 meter
        water_density = float(
            get_physical_constant(
                'relic.physical.constants.water.density_liquid'
            )
        )
        
        mass_content = UnitConverter.soil_moisture_m3m3_to_kgm2(
            volume_fraction, soil_depth, water_density
        )
        
        expected = volume_fraction * soil_depth * water_density
        torch.testing.assert_close(mass_content, expected)
        
        # Test reverse conversion
        recovered_volume = UnitConverter.soil_moisture_kgm2_to_m3m3(
            mass_content, soil_depth, water_density
        )
        
        torch.testing.assert_close(recovered_volume, volume_fraction)
    
    def test_soil_moisture_variable_depth(self):
        """Test soil moisture conversion with variable soil depth."""
        volume_fraction = torch.tensor([0.25])
        soil_depth = torch.tensor([0.5, 1.0, 2.0])  # Variable depths
        water_density = get_physical_constant(
            "relic.physical.constants.water.density_liquid"
        )

        mass_content = UnitConverter.soil_moisture_m3m3_to_kgm2(
            volume_fraction, soil_depth, water_density
        )
        
        expected = volume_fraction * soil_depth * water_density
        torch.testing.assert_close(mass_content, expected)
    
    def test_ensure_tensor_utility(self):
        """Test tensor conversion utility function."""
        # Test with float conversion
        from relic.core.functor import _ensure_tensor
        
        result = _ensure_tensor(5.0)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 5.0
        
        # Test with existing tensor
        input_tensor = torch.tensor([1.0, 2.0])
        result = _ensure_tensor(input_tensor)
        assert torch.equal(result, input_tensor)
        
        cpu_device = torch.device('cpu')
        result = _ensure_tensor(5.0, device=cpu_device)
        assert result.device == cpu_device


class TestBridgeIntegration:
    """Integration tests for bridge classes working together."""
    
    def test_multiple_bridges_different_categories(self):
        """Test that different bridge types have correct categories."""
        
        class TestBio(BiologicalProcessFunctor):
            def get_dependencies(self): return []
            def forward(self, state, time, dt, **kwargs): return {}
        
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def get_dependencies(self): return []
            def forward(self, state, time, dt, **kwargs): return {}
        
        class TestField(PhysicalFieldFunctor):
            SPATIAL_DIMS = None

            def get_dependencies(self): return []
            def forward(self, state, time, dt, **kwargs): return {}
            def get_value_at(self, coordinates):
                raise NotImplementedError("point model (SPATIAL_DIMS = None): no coordinate query")
        
        class TestCoupling(ScaleCouplingFunctor):
            def get_dependencies(self): return []
            def forward(self, state, time, dt, **kwargs): return {}
        
        bio = TestBio(
            name='bio',
            config={},
            metadata=_metadata(
                'bio',
                ProcessCategory.BIOLOGICAL_PROCESS,
                inputs=('fixture_state',),
            ),
        )
        transport = TestTransport(
            name='transport',
            config={},
            metadata=_metadata(
                'transport',
                ProcessCategory.TRANSPORT_MECHANISM,
                inputs=('fixture_state',),
            ),
        )
        field = TestField(
            name='field',
            config={},
            metadata=_metadata(
                'field',
                ProcessCategory.PHYSICAL_FIELD,
                inputs=('fixture_state',),
            ),
        )
        coupling = TestCoupling(
            name='coupling',
            config={},
            metadata=_metadata(
                'coupling',
                ProcessCategory.SCALE_COUPLING,
                inputs=('fixture_state',),
            ),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        )
        
        # Test category assignment
        assert bio.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS
        assert transport.metadata.category == ProcessCategory.TRANSPORT_MECHANISM  
        assert field.metadata.category == ProcessCategory.PHYSICAL_FIELD
        assert coupling.metadata.category == ProcessCategory.SCALE_COUPLING
        
        # Test they are all FunctorBase instances
        from relic.core.functor import FunctorBase
        
        assert isinstance(bio, FunctorBase)
        assert isinstance(transport, FunctorBase)
        assert isinstance(field, FunctorBase)
        assert isinstance(coupling, FunctorBase)
    
    def test_device_consistency(self):
        """Test the documented pre-registration CPU placeholder."""
        
        class TestProcess(BiologicalProcessFunctor):
            def get_dependencies(self): 
                return []
                
            def forward(self, state, time, dt, **kwargs): 
                # Return tensor on CPU since device property might not be available
                return {'rate': torch.tensor(1.0)}
        
        # Test basic functionality without specific device requirements
        cpu_functor = TestProcess(
            name='cpu_test',
            config={},
            metadata=_metadata(
                'cpu_test',
                ProcessCategory.BIOLOGICAL_PROCESS,
                outputs=('rate',),
            ),
        )
        cpu_result = cpu_functor.forward({}, 0, 1.0)
        assert cpu_functor.device == torch.device('cpu')
        assert isinstance(cpu_result['rate'], torch.Tensor)
        assert cpu_result['rate'].device == torch.device('cpu')
        assert cpu_result['rate'].item() == 1.0

if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])

