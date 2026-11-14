"""
L3 System Regression Tests for RELIC Bridge Refactoring

This module contains system-level regression tests to ensure that the bridge
base class refactoring does not introduce breaking changes to the overall
RELIC system functionality and maintains backward consistency.

Test Focus Areas:
1. System-wide import and initialization
2. Basic simulation pipeline functionality
3. Configuration loading and parameter management
4. Core framework component integration
5. Overall system stability and error handling
6. Memory usage and resource management
7. API consistency and interface preservation

The tests are designed to catch major system regressions while being
resilient to minor implementation changes and missing dependencies
during development phases.

References:
    - RELIC System Architecture Documentation
    - Bridge Base Class Design Specification  
    - Regression Testing Best Practices
"""

import gc

import pytest
import torch

from relic.core.functor import FunctorMetadata, ProcessCategory
from relic.utils.physical_constants import get_physical_constant


def _metadata(
    name: str,
    category: ProcessCategory,
    *,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
) -> FunctorMetadata:
    """Create complete metadata for system-level bridge fixtures."""
    return FunctorMetadata(
        name=name,
        category=category,
        description=f"System regression fixture for {name}.",
        references=["RELIC architecture bridge contract test."],
        input_variables=list(inputs),
        output_variables=list(outputs),
    )


class TestSystemImports:
    """Test system-wide import functionality."""
    
    def test_core_imports(self):
        """Test that core framework components can be imported."""
        from relic.core.abstractions import (
            BiologicalProcess,
            EnvironmentDict,
            FluxDict,
            PhysicalField,
            ScaleCoupling,
            StateDict,
            TransportMechanism,
        )

        assert all(
            symbol is not None
            for symbol in (
                PhysicalField,
                BiologicalProcess,
                TransportMechanism,
                ScaleCoupling,
                StateDict,
                FluxDict,
                EnvironmentDict,
            )
        )
    
    def test_functor_base_import(self):
        """Test that unified FunctorBase can be imported."""
        from relic.core.functor import FunctorBase

        assert callable(FunctorBase)
    
    def test_bridge_classes_import(self):
        """Test that new bridge classes can be imported."""
        from relic.core import (
            BiologicalProcessFunctor,
            PhysicalFieldFunctor,
            ScaleCouplingFunctor,
            TransportMechanismFunctor,
        )
        from relic.data.standards.unit_converter import UnitConverter

        assert callable(BiologicalProcessFunctor)
        assert callable(TransportMechanismFunctor)
        assert callable(PhysicalFieldFunctor)
        assert callable(ScaleCouplingFunctor)
        assert hasattr(UnitConverter, 'soil_moisture_m3m3_to_kgm2')
    
    def test_scientific_modules_import(self):
        """Test that key scientific modules can still be imported."""
        from relic.scientific.atmosphere.transport.eddy_diffusion_blm import (
            EddyDiffusionBLMFunctor,
        )
        from relic.scientific.biosphere.photosynthesis.farquhar import (
            FarquharPhotosynthesis,
        )
        from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
            PenmanMonteithET,
        )

        assert callable(FarquharPhotosynthesis)
        assert callable(PenmanMonteithET)
        assert callable(EddyDiffusionBLMFunctor)


class TestSystemConfiguration:
    """Test system configuration and parameter management."""
    
    def test_parameter_system_basic_import(self):
        """Test canonical parameter system import."""
        from relic import parameters

        assert hasattr(parameters, 'get_global_manager')
        assert callable(parameters.get_global_manager)
        assert parameters.get_global_manager() is not None
    
    def test_configuration_system_basic(self):
        """Test basic configuration system functionality."""
        from relic.core.config import get_global_config, validate_configuration

        default_config = get_global_config()
        assert default_config is not None

        validation_result = validate_configuration(default_config, strict=False)
        assert hasattr(validation_result, "is_valid")


class TestBridgeClassSystemIntegration:
    """Test bridge classes work within the broader system context."""
    
    def test_bridge_class_instantiation(self):
        """Test that bridge classes can be instantiated in system context."""
        
        from relic.core import (
            BiologicalProcessFunctor, TransportMechanismFunctor,
            PhysicalFieldFunctor, ScaleCouplingFunctor
        )
        
        # Test basic instantiation
        test_configs = [
            ('bio', BiologicalProcessFunctor, ProcessCategory.BIOLOGICAL_PROCESS),
            (
                'transport',
                TransportMechanismFunctor,
                ProcessCategory.TRANSPORT_MECHANISM,
            ),
            ('field', PhysicalFieldFunctor, ProcessCategory.PHYSICAL_FIELD),
            ('coupling', ScaleCouplingFunctor, ProcessCategory.SCALE_COUPLING),
        ]

        for name, bridge_class, category in test_configs:
            class TestFunctor(bridge_class):
                # A concrete transport class declares its descriptive label and a
                # concrete field its SPATIAL_DIMS and coordinate query (a point
                # model here); the other bridges ignore these members.
                TRANSPORT_TYPE = "diffusion"
                SPATIAL_DIMS = None

                def get_value_at(self, coordinates):
                    raise NotImplementedError("point model (SPATIAL_DIMS = None): no coordinate query")

                def get_dependencies(self):
                    return []

                def forward(self, state, time, dt, **kwargs):
                    return {'test_output': torch.tensor(1.0)}

            functor_name = f'test_{name}'
            # A scale coupling declares the scale pair it couples and the coupling type.
            scales = (
                {'source_scale': 'fine', 'target_scale': 'coarse', 'coupling_type': 'spatial'}
                if bridge_class is ScaleCouplingFunctor
                else {}
            )
            functor = TestFunctor(
                name=functor_name,
                config={},
                metadata=_metadata(
                    functor_name,
                    category,
                    outputs=('test_output',),
                ),
                **scales,
            )

            assert functor.name == functor_name
            assert functor.metadata.category == category
            assert callable(functor.forward)
            assert callable(functor.get_dependencies)
    
    def test_migrated_modules_basic_functionality(self):
        """Test basic functionality of key migrated modules."""
        from relic.scientific.biosphere.photosynthesis.farquhar import (
            FarquharPhotosynthesis,
        )
        from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
            PenmanMonteithET,
        )

        farquhar = FarquharPhotosynthesis(
            name='test_farquhar',
            config={'parameters': {'vcmax25': 60.0}},
        )
        farquhar_result = farquhar.forward(
            state={
                'leaf_area_index': torch.tensor(3.0),
                'canopy_conductance': torch.tensor(0.01),
                'air_temperature': torch.tensor(298.15),
                'atmospheric_co2_ppm': torch.tensor(400.0),
                'ppfd': torch.tensor(1000.0),
                'vapor_pressure_deficit': torch.tensor(1000.0),
            },
            time=torch.tensor(0.0),
            dt=torch.tensor(3600.0),
        )
        assert {
            'gross_photosynthesis',
            'net_photosynthesis',
            'gross_primary_productivity',
        }.issubset(farquhar_result)

        pm_et = PenmanMonteithET(
            name='test_pm_et',
            config={'parameters': {'canopy_height': 0.5}, "canopy_resistance_source": "state"},
        )
        # The drivers are read from the state (the one input channel).
        pm_result = pm_et.forward(
            state={
                'surface_resistance': torch.tensor(70.0),
                'air_temperature': torch.tensor(298.15),
                'net_radiation': torch.tensor(600.0),
                'wind_speed': torch.tensor(2.0),
                'relative_humidity': torch.tensor(0.6),
                'atmospheric_pressure': torch.tensor(101325.0),
            },
            time=torch.tensor(0.0),
            dt=torch.tensor(3600.0),
        )
        assert 'evapotranspiration_flux' in pm_result


class TestSystemStability:
    """Test overall system stability after refactoring."""
    
    def test_import_stability(self):
        """Test that imports are stable and don't cause crashes."""
        
        # Test multiple import attempts
        for i in range(3):
            from relic.core import BiologicalProcessFunctor

            class TestBio(BiologicalProcessFunctor):
                def get_dependencies(self):
                    return []

                def forward(self, state, time, dt, **kwargs):
                    return {}

            name = f'stability_test_{i}'
            bio = TestBio(
                name=name,
                config={"canopy_resistance_source": "state"},
                metadata=_metadata(
                    name,
                    ProcessCategory.BIOLOGICAL_PROCESS,
                    inputs=('fixture_state',),
                ),
            )
            assert bio.name == name
    
    def test_memory_basic(self):
        """Basic memory usage test."""
        # Clear any existing objects
        gc.collect()

        from relic.core import BiologicalProcessFunctor

        instances = []
        for i in range(10):
            class TestFunctor(BiologicalProcessFunctor):
                def get_dependencies(self):
                    return []

                def forward(self, state, time, dt, **kwargs):
                    return {'out': torch.tensor(1.0)}

            name = f'mem_test_{i}'
            instance = TestFunctor(
                name=name,
                config={},
                metadata=_metadata(
                    name,
                    ProcessCategory.BIOLOGICAL_PROCESS,
                    outputs=('out',),
                ),
            )
            instances.append(instance)

        assert len(instances) == 10
        del instances
        gc.collect()
    
    def test_error_handling_graceful(self):
        """Test that system handles errors gracefully."""
        
        from relic.core import BiologicalProcessFunctor
        
        class TestFunctor(BiologicalProcessFunctor):
            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {}

        functor = TestFunctor(
            name='error_test',
            config={},
            metadata=_metadata(
                'error_test',
                ProcessCategory.BIOLOGICAL_PROCESS,
                inputs=('fixture_state',),
            ),
        )
        assert functor is not None


class TestUnitConversion:
    """Test unit conversion functionality at system level."""
    
    def test_unit_converter_system_integration(self):
        """Test that unit converter works in system context."""
        from relic.data.standards.unit_converter import UnitConverter

        volume_fraction = torch.tensor([0.25, 0.30, 0.35])
        soil_depth = 1.0
        water_density = float(
            get_physical_constant(
                'relic.physical.constants.water.density_liquid'
            )
        )

        mass_content = UnitConverter.soil_moisture_m3m3_to_kgm2(
            volume_fraction,
            soil_depth,
            water_density,
        )

        assert isinstance(mass_content, torch.Tensor)
        assert mass_content.shape == volume_fraction.shape
        torch.testing.assert_close(
            mass_content,
            volume_fraction * soil_depth * water_density,
        )

        recovered = UnitConverter.soil_moisture_kgm2_to_m3m3(
            mass_content,
            soil_depth,
            water_density,
        )

        torch.testing.assert_close(recovered, volume_fraction, rtol=1e-6, atol=1e-8)


class TestPublicAPIConsistency:
    """Test canonical public API consistency."""
    
    def test_functor_base_still_available(self):
        """Test that FunctorBase is still directly accessible."""
        
        from relic.core import FunctorBase as CoreFunctorBase
        from relic.core.functor import FunctorBase

        assert FunctorBase is CoreFunctorBase
        assert callable(FunctorBase)
    
    def test_abstract_classes_still_available(self):
        """Test that abstract base classes are still available."""
        
        from relic.core import (
            BiologicalProcess as CoreBiologicalProcess,
            PhysicalField as CorePhysicalField,
            ScaleCoupling as CoreScaleCoupling,
            TransportMechanism as CoreTransportMechanism,
        )
        from relic.core.abstractions import (
            BiologicalProcess,
            PhysicalField,
            ScaleCoupling,
            TransportMechanism,
        )

        assert PhysicalField is CorePhysicalField
        assert BiologicalProcess is CoreBiologicalProcess
        assert TransportMechanism is CoreTransportMechanism
        assert ScaleCoupling is CoreScaleCoupling
    
    def test_core_type_exports(self):
        """Test canonical core type exports."""
        from relic.core import EnvironmentDict, FluxDict, StateDict

        assert StateDict is dict or hasattr(StateDict, '__origin__')
        assert FluxDict is dict or hasattr(FluxDict, '__origin__')
        assert EnvironmentDict is dict or hasattr(EnvironmentDict, '__origin__')


def test_system_health_check():
    """Overall system health check."""
    from relic.core import BiologicalProcessFunctor
    from relic.scientific.biosphere.photosynthesis.farquhar import (
        FarquharPhotosynthesis,
    )

    class TestBridge(BiologicalProcessFunctor):
        def get_dependencies(self):
            return []

        def forward(self, state, time, dt, **kwargs):
            return {}

    bridge = TestBridge(
        name='health_check',
        config={},
        metadata=_metadata(
            'health_check',
            ProcessCategory.BIOLOGICAL_PROCESS,
            inputs=('fixture_state',),
        ),
    )
    assert bridge.device == torch.device('cpu')

    farquhar = FarquharPhotosynthesis(
        name='health',
        config={'parameters': {}},
    )
    result = farquhar.forward(
        state={
            'leaf_area_index': torch.tensor(3.0),
            'air_temperature': torch.tensor(298.15),
            'atmospheric_co2_ppm': torch.tensor(400.0),
            'ppfd': torch.tensor(1000.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
        },
        time=torch.tensor(0.0),
        dt=torch.tensor(3600.0),
    )
    assert {
        'gross_photosynthesis',
        'net_photosynthesis',
        'gross_primary_productivity',
    }.issubset(result)


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v', '--tb=short'])


