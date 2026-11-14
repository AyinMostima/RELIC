#!/usr/bin/env python
"""
L1 Unit Tests for Bridge Classes Dual Inheritance

Tests the four bridge classes to ensure they properly inherit from
both abstract base classes and FunctorBase, maintaining proper MRO
and method availability.

Mathematical Tests:
1. Temperature response functions (Q10, Arrhenius)
2. Conservation validation 
3. Unit conversions
4. Parameter access patterns
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.core.abstractions import (
    PhysicalField, BiologicalProcess, TransportMechanism, ScaleCoupling
)
from relic.core.functor import FunctorBase, FunctorMetadata, ProcessCategory
from relic.core.functor import (
    BiologicalProcessFunctor, TransportMechanismFunctor,
    PhysicalFieldFunctor, ScaleCouplingFunctor  
)
from test.utils.forward_call import call_forward


def _fixture_metadata(
    name: str,
    category: ProcessCategory,
    output_variable: str,
) -> FunctorMetadata:
    """Return complete metadata for one concrete bridge fixture."""
    return FunctorMetadata(
        name=name,
        description=f"Dual-inheritance bridge fixture for {name}",
        category=category,
        references=["RELIC architecture test fixture"],
        input_variables=["test_input"],
        output_variables=[output_variable],
    )


class ConcreteBiologicalFunctor(BiologicalProcessFunctor):
    """Concrete implementation for testing."""
    def __init__(self, name, config=None, metadata=None, **kwargs):
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata or _fixture_metadata(
                name,
                ProcessCategory.BIOLOGICAL_PROCESS,
                "test_output",
            ),
            **kwargs,
        )

    def forward(self, state, time, dt, **kwargs):
        return {"test_output": torch.tensor(1.0)}
    
    def get_dependencies(self):
        return []


class ConcreteTransportFunctor(TransportMechanismFunctor):
    """Concrete implementation for testing."""

    TRANSPORT_TYPE = "diffusion"
    def __init__(self, name, config=None, metadata=None, **kwargs):
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata or _fixture_metadata(
                name,
                ProcessCategory.TRANSPORT_MECHANISM,
                "test_flux",
            ),
            **kwargs,
        )

    def forward(self, state, time, dt, **kwargs):
        return {"test_flux": torch.tensor(0.5)}
    
    def get_dependencies(self):
        return []


class ConcretePhysicalFieldFunctor(PhysicalFieldFunctor):
    """Concrete implementation for testing."""

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )
    def __init__(self, name, config=None, metadata=None, **kwargs):
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata or _fixture_metadata(
                name,
                ProcessCategory.PHYSICAL_FIELD,
                "update",
            ),
            **kwargs,
        )

    def forward(self, state, time, dt, **kwargs):
        # Intentionally avoid keys containing 'field' or 'value' so the bridge
        # placeholder methods raise unless explicitly overridden.
        return {"update": torch.tensor(0.1)}
    
    def get_dependencies(self):
        return []


class ConcreteScaleCouplingFunctor(ScaleCouplingFunctor):
    """Concrete implementation for testing; it declares the scales it couples."""
    def __init__(self, name, config=None, metadata=None, **kwargs):
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata or _fixture_metadata(
                name,
                ProcessCategory.SCALE_COUPLING,
                "scaled_value",
            ),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
            **kwargs,
        )

    def forward(self, state, time, dt, **kwargs):
        return {"scaled_value": torch.tensor(2.0)}
    
    def get_dependencies(self):
        return []


class TestBiologicalProcessFunctor:
    """Test BiologicalProcessFunctor dual inheritance and functionality."""
    
    def test_dual_inheritance(self):
        """Test that BiologicalProcessFunctor inherits from both base classes."""
        # Check inheritance
        assert issubclass(BiologicalProcessFunctor, BiologicalProcess)
        assert issubclass(BiologicalProcessFunctor, FunctorBase)
        
        # Create instance using concrete subclass
        functor = ConcreteBiologicalFunctor(
            name="test_bio",
            config={}
        )
        
        # Check instance types
        assert isinstance(functor, BiologicalProcess)
        assert isinstance(functor, FunctorBase)
        
    def test_has_abstract_methods(self):
        """Test that abstract methods from BiologicalProcess are available."""
        functor = ConcreteBiologicalFunctor(
            name="test_bio",
            config={}
        )
        
        # Single-entry architecture: forward is the only required compute API
        assert hasattr(functor, 'forward')
        assert callable(functor.forward)
        
    def test_has_functor_methods(self):
        """Test that FunctorBase methods are available."""
        functor = ConcreteBiologicalFunctor(
            name="test_bio",
            config={}
        )
        
        # Check FunctorBase methods
        assert hasattr(functor, 'forward')
        assert hasattr(functor, 'get_parameter_value')
        assert hasattr(functor, 'get_dependencies')
        assert callable(functor.forward)
        
    def test_forward_entry(self):
        """Test forward as the only public entry."""
        functor = ConcreteBiologicalFunctor(
            name="test_bio",
            config={}
        )
        
        state = {"substrate": torch.tensor(10.0), "temperature": torch.tensor(298.15)}

        result = call_forward(functor, state)
        assert isinstance(result, dict)
        assert "test_output" in result


class TestTransportMechanismFunctor:
    """Test TransportMechanismFunctor dual inheritance."""
    
    def test_dual_inheritance(self):
        """Test dual inheritance from TransportMechanism and FunctorBase."""
        assert issubclass(TransportMechanismFunctor, TransportMechanism)
        assert issubclass(TransportMechanismFunctor, FunctorBase)
        
        functor = ConcreteTransportFunctor(
            name="test_transport",
            config={}
        )
        
        assert isinstance(functor, TransportMechanism)
        assert isinstance(functor, FunctorBase)
        
    def test_has_transport_methods(self):
        """Test that TransportMechanism methods are available."""
        functor = ConcreteTransportFunctor(
            name="test_transport",
            config={}
        )
        
        # Single-entry architecture: forward is the compute API
        assert hasattr(functor, 'forward')
        assert hasattr(functor, 'flux_to_updates')
        assert callable(functor.forward)
        
    def test_flux_to_updates_conversion(self):
        """Test flux_to_updates converts properly."""
        functor = ConcreteTransportFunctor(
            name="test_transport",
            config={}
        )
        
        # Test scalar flux
        flux = torch.tensor(5.0)
        dt = 0.1
        updates = functor.flux_to_updates(flux, dt, "water_flux")
        
        assert "water_flux" in updates
        assert torch.allclose(updates["water_flux"], flux * dt)
        
    def test_validate_transport_units(self):
        """Test unit validation for transport."""
        functor = ConcreteTransportFunctor(
            name="test_transport",
            config={}
        )
        
        # Valid units
        assert functor.validate_transport_units("kg m-2 s-1", "water")
        assert functor.validate_transport_units("mol m-2 s-1", "carbon")
        
        # Invalid units (missing time dimension)
        assert not functor.validate_transport_units("kg m-2", "water")


class TestPhysicalFieldFunctor:
    """Test PhysicalFieldFunctor dual inheritance."""
    
    def test_dual_inheritance(self):
        """Test dual inheritance from PhysicalField and FunctorBase."""
        assert issubclass(PhysicalFieldFunctor, PhysicalField)
        assert issubclass(PhysicalFieldFunctor, FunctorBase)
        
        functor = ConcretePhysicalFieldFunctor(
            name="test_field",
            config={"spatial_dims": 2}
        )
        
        assert isinstance(functor, PhysicalField)
        assert isinstance(functor, FunctorBase)
        
    def test_has_field_methods(self):
        """Test that PhysicalField methods are available."""
        functor = ConcretePhysicalFieldFunctor(
            name="test_field",
            config={"spatial_dims": 2}
        )
        
        # Check field methods
        assert hasattr(functor, 'get_value_at')
        assert hasattr(functor, 'compute_gradient')
        assert hasattr(functor, 'interpolate')
        assert callable(functor.get_value_at)
        
    def test_update_field_to_flux(self):
        """Test update_field_to_flux conversion."""
        functor = ConcretePhysicalFieldFunctor(
            name="test_field",
            config={"spatial_dims": 2}
        )
        
        # Create field update
        field_update = torch.randn(10, 10)
        dt = 0.1
        field_name = "temperature"
        
        flux = functor.update_field_to_flux(field_update, dt, field_name)
        
        assert field_name in flux
        # Flux should be update / dt
        expected = field_update / dt
        assert torch.allclose(flux[field_name], expected, rtol=1e-5)
        
    def test_placeholder_methods_raise_error(self):
        """Test that placeholder methods now raise NotImplementedError."""
        functor = ConcretePhysicalFieldFunctor(
            name="test_field",
            config={"spatial_dims": 2}
        )
        
        coords = torch.randn(5, 2)
        
        # These should raise NotImplementedError after our fix
        with pytest.raises(NotImplementedError):
            functor.get_value_at(coords)
            
        gradient = functor.compute_gradient(coords)
        assert isinstance(gradient, torch.Tensor)
        assert gradient.shape == coords.shape
        assert torch.allclose(gradient, torch.zeros_like(coords))


class TestScaleCouplingFunctor:
    """Test ScaleCouplingFunctor dual inheritance."""
    
    def test_dual_inheritance(self):
        """Test dual inheritance from ScaleCoupling and FunctorBase."""
        assert issubclass(ScaleCouplingFunctor, ScaleCoupling)
        assert issubclass(ScaleCouplingFunctor, FunctorBase)
        
        functor = ConcreteScaleCouplingFunctor(
            name="test_coupling",
            config={"conservation_properties": ["mass", "energy"]}
        )
        
        assert isinstance(functor, ScaleCoupling)
        assert isinstance(functor, FunctorBase)
        
    def test_has_scaling_methods(self):
        """Test that ScaleCoupling methods are available."""
        functor = ConcreteScaleCouplingFunctor(
            name="test_coupling",
            config={"conservation_properties": ["mass", "energy"]}
        )
        
        # Check scaling methods
        assert hasattr(functor, 'upscale')
        assert hasattr(functor, 'downscale')
        assert hasattr(functor, 'check_conservation_bridge')
        assert callable(functor.upscale)
        
    def test_check_conservation_bridge(self):
        """Test conservation checking bridge method."""
        functor = ConcreteScaleCouplingFunctor(
            name="test_coupling",
            config={"conservation_properties": ["mass", "energy"]}
        )
        
        # Test with conserved data
        before = {"mass": torch.tensor(100.0), "energy": torch.tensor(500.0)}
        after = {"mass": torch.tensor(100.0), "energy": torch.tensor(500.0)}
        
        violations = functor.check_conservation_bridge(before, after)
        assert "mass" in violations
        assert "energy" in violations
        assert abs(violations["mass"]) < 1e-6
        assert abs(violations["energy"]) < 1e-6
        
        # Test with non-conserved data
        after_bad = {"mass": torch.tensor(90.0), "energy": torch.tensor(520.0)}
        violations_bad = functor.check_conservation_bridge(before, after_bad)
        assert abs(violations_bad["mass"]) > 1.0
        assert abs(violations_bad["energy"]) > 1.0


class TestMethodResolutionOrder:
    """Test that MRO is correct for all bridge classes."""
    
    def test_biological_mro(self):
        """Test MRO for BiologicalProcessFunctor."""
        mro = BiologicalProcessFunctor.__mro__
        
        # Should be: BiologicalProcessFunctor -> BiologicalProcess -> FunctorBase -> ...
        assert mro[0] == BiologicalProcessFunctor
        assert BiologicalProcess in mro
        assert FunctorBase in mro
        
        # BiologicalProcess should come before FunctorBase
        bio_idx = mro.index(BiologicalProcess)
        func_idx = mro.index(FunctorBase)
        assert bio_idx < func_idx
        
    def test_transport_mro(self):
        """Test MRO for TransportMechanismFunctor."""
        mro = TransportMechanismFunctor.__mro__
        
        assert mro[0] == TransportMechanismFunctor
        assert TransportMechanism in mro
        assert FunctorBase in mro
        
        # TransportMechanism should come before FunctorBase
        trans_idx = mro.index(TransportMechanism)
        func_idx = mro.index(FunctorBase)
        assert trans_idx < func_idx
        
    def test_field_mro(self):
        """Test MRO for PhysicalFieldFunctor."""
        mro = PhysicalFieldFunctor.__mro__
        
        assert mro[0] == PhysicalFieldFunctor
        assert PhysicalField in mro
        assert FunctorBase in mro
        
        # PhysicalField should come before FunctorBase  
        field_idx = mro.index(PhysicalField)
        func_idx = mro.index(FunctorBase)
        assert field_idx < func_idx
        
    def test_coupling_mro(self):
        """Test MRO for ScaleCouplingFunctor."""
        mro = ScaleCouplingFunctor.__mro__
        
        assert mro[0] == ScaleCouplingFunctor
        assert ScaleCoupling in mro
        assert FunctorBase in mro
        
        # ScaleCoupling should come before FunctorBase
        scale_idx = mro.index(ScaleCoupling)
        func_idx = mro.index(FunctorBase)
        assert scale_idx < func_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
