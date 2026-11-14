"""
Test suite for bridge classes with dual inheritance architecture.

This test verifies that the bridge classes correctly implement both:
1. Abstract base class interfaces (PhysicalField, BiologicalProcess, etc.)
2. FunctorBase system interaction layer

Tests ensure proper MRO (Method Resolution Order), abstract method implementation,
correct initialization of both abstract and functor layers, and proper attribute inheritance.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add relic to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.core.functor import (
    BiologicalProcessFunctor,
    TransportMechanismFunctor,
    PhysicalFieldFunctor,
    ScaleCouplingFunctor
)
from relic.core.abstractions import (
    PhysicalField,
    BiologicalProcess,
    TransportMechanism,
    ScaleCoupling
)
from relic.core.functor import FunctorBase, FunctorMetadata, ProcessCategory
from test.utils.forward_call import call_forward, call_transport_forward


def _test_metadata(name: str, category: ProcessCategory) -> FunctorMetadata:
    """Build complete metadata for bridge test fixtures."""
    return FunctorMetadata(
        name=name,
        description=f"Bridge test fixture for {name}",
        category=category,
        references=["RELIC architecture test fixture"],
        input_variables=["test_input"],
        output_variables=["test_output"],
    )


class TestBridgeClassInheritance(unittest.TestCase):
    """Test that bridge classes correctly inherit from both abstract classes and FunctorBase."""

    def test_biological_process_functor_inheritance(self):
        """Verify BiologicalProcessFunctor inherits from both BiologicalProcess and FunctorBase."""
        assert issubclass(BiologicalProcessFunctor, BiologicalProcess)
        assert issubclass(BiologicalProcessFunctor, FunctorBase)

        # Check MRO includes both parent classes
        mro = BiologicalProcessFunctor.__mro__
        assert BiologicalProcess in mro
        assert FunctorBase in mro

    def test_transport_mechanism_functor_inheritance(self):
        """Verify TransportMechanismFunctor inherits from both TransportMechanism and FunctorBase."""
        assert issubclass(TransportMechanismFunctor, TransportMechanism)
        assert issubclass(TransportMechanismFunctor, FunctorBase)

        mro = TransportMechanismFunctor.__mro__
        assert TransportMechanism in mro
        assert FunctorBase in mro

    def test_physical_field_functor_inheritance(self):
        """Verify PhysicalFieldFunctor inherits from both PhysicalField and FunctorBase."""
        assert issubclass(PhysicalFieldFunctor, PhysicalField)
        assert issubclass(PhysicalFieldFunctor, FunctorBase)

        mro = PhysicalFieldFunctor.__mro__
        assert PhysicalField in mro
        assert FunctorBase in mro

    def test_scale_coupling_functor_inheritance(self):
        """Verify ScaleCouplingFunctor inherits from both ScaleCoupling and FunctorBase."""
        assert issubclass(ScaleCouplingFunctor, ScaleCoupling)
        assert issubclass(ScaleCouplingFunctor, FunctorBase)

        mro = ScaleCouplingFunctor.__mro__
        assert ScaleCoupling in mro
        assert FunctorBase in mro


class TestBridgeClassInitialization(unittest.TestCase):
    """Test that bridge classes properly initialize both abstract and functor layers."""

    def test_biological_process_initialization(self):
        """Test BiologicalProcessFunctor initializes with correct attributes from both layers."""

        class TestBioFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {'test_rate': torch.tensor(1.0)}

            def get_dependencies(self):
                return []

        functor = TestBioFunctor(
            name='test_bio',
            metadata=_test_metadata('test_bio', ProcessCategory.BIOLOGICAL_PROCESS),
            process_name='test_process',
            allometric_scaling=0.75
        )

        # Test abstract base class attributes are initialized
        assert hasattr(functor, 'process_name')
        assert functor.process_name == 'test_process'
        assert functor.allometric_scaling == 0.75

        # Test FunctorBase attributes are initialized
        assert hasattr(functor, 'name')
        assert functor.name == 'test_bio'
        assert hasattr(functor, 'metadata')
        assert functor.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS

    def test_transport_mechanism_initialization(self):
        """Test TransportMechanismFunctor initializes with correct attributes from both layers."""

        class TestTransportFunctor(TransportMechanismFunctor):
            TRANSPORT_TYPE = "advection"

            def forward(self, state, time, dt, **kwargs):
                return {'test_flux': torch.tensor(0.5)}

            def get_dependencies(self):
                return []

        functor = TestTransportFunctor(
            name='test_transport',
            metadata=_test_metadata(
                'test_transport',
                ProcessCategory.TRANSPORT_MECHANISM,
            ),
            peclet_number=100.0
        )
        functor.set_scalar_diffusivity(2.5e-5)

        # Test abstract base class attributes
        assert hasattr(functor, 'transport_type')
        assert functor.transport_type == 'advection'
        assert functor.diffusivity == 2.5e-5
        assert functor.peclet_number == 100.0

        # Test FunctorBase attributes
        assert functor.name == 'test_transport'
        assert functor.metadata.category == ProcessCategory.TRANSPORT_MECHANISM

    def test_physical_field_initialization(self):
        """Test PhysicalFieldFunctor initializes with correct attributes from both layers."""

        class TestFieldFunctor(PhysicalFieldFunctor):
            SPATIAL_DIMS = 3

            def forward(self, state, time, dt, **kwargs):
                return {'field_update': torch.zeros(10)}

            def get_dependencies(self):
                return []

            def get_value_at(self, coordinates):
                return coordinates[..., 0]

        functor = TestFieldFunctor(
            name='temperature_field',
            metadata=_test_metadata(
                'temperature_field',
                ProcessCategory.PHYSICAL_FIELD,
            ),
        )

        # Abstract base class layer: SPATIAL_DIMS is the one declared field
        # property; no field identity, units, initial value or time flag.
        assert functor.SPATIAL_DIMS == 3
        for removed in ('field_name', 'units', 'default_value', 'spatial_dims', 'temporal_dependence'):
            assert not hasattr(functor, removed), removed

        # Test FunctorBase attributes
        assert functor.name == 'temperature_field'
        assert functor.metadata.category == ProcessCategory.PHYSICAL_FIELD

    def test_scale_coupling_initialization(self):
        """Test ScaleCouplingFunctor initializes with correct attributes from both layers."""

        class TestScaleFunctor(ScaleCouplingFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {'scaled_data': torch.tensor(1.5)}

            def get_dependencies(self):
                return []

        functor = TestScaleFunctor(
            name='test_scale',
            metadata=_test_metadata('test_scale', ProcessCategory.SCALE_COUPLING),
            source_scale='leaf',
            target_scale='canopy',
            coupling_type='hierarchical',
            conservation_properties=['mass', 'energy']
        )

        # Test abstract base class attributes
        assert hasattr(functor, 'source_scale')
        assert functor.source_scale == 'leaf'
        assert functor.target_scale == 'canopy'
        assert functor.coupling_type == 'hierarchical'
        assert functor.conservation_properties == ['mass', 'energy']

        # Test FunctorBase attributes
        assert functor.name == 'test_scale'
        assert functor.metadata.category == ProcessCategory.SCALE_COUPLING


class TestAbstractMethodImplementation(unittest.TestCase):
    """Test that bridge classes provide implementations for abstract methods."""

    def test_biological_process_forward(self):
        """Test BiologicalProcessFunctor computes rates via forward()."""

        class TestBioFunctor(BiologicalProcessFunctor):
            def temperature_function(self, temp):
                # The concrete Functor owns its law: Q10 = 2 about 298.15 K per 10 K.
                return 2.0 ** ((temp - 298.15) / 10.0)

            def forward(self, state, time, dt, **kwargs):
                temp = state["temperature"]
                conc = state["concentration"]
                return {"test_rate": conc * self.temperature_function(temp)}

            def get_dependencies(self):
                return []

        functor = TestBioFunctor(
            name='test_bio',
            metadata=_test_metadata('test_bio', ProcessCategory.BIOLOGICAL_PROCESS),
        )

        # Test forward method works
        state = {'concentration': torch.tensor(2.0), 'temperature': torch.tensor(298.15)}

        result = call_forward(functor, state)
        assert isinstance(result, dict)
        assert "rate" in result
        assert isinstance(result["rate"], torch.Tensor)
        assert result["rate"].item() > 0  # Should be positive

    def test_transport_mechanism_forward(self):
        """Test TransportMechanismFunctor computes fluxes via forward()."""

        class TestTransportFunctor(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                source = state.get("source", torch.tensor(1.0))
                sink = state.get("sink", torch.tensor(0.0))
                distance = state.get("distance", torch.tensor(1.0))
                gradient = (source - sink) / distance
                return {"transport_flux": -self.diffusivity * gradient}

            def get_dependencies(self):
                return []

        functor = TestTransportFunctor(
            name='test_transport',
            metadata=_test_metadata(
                'test_transport',
                ProcessCategory.TRANSPORT_MECHANISM,
            ),
        )
        functor.set_scalar_diffusivity(1e-5)

        # Test forward method
        source = torch.tensor(2.0)
        sink = torch.tensor(1.0)
        distance = torch.tensor(0.01)

        result = call_transport_forward(functor, source, sink, distance=distance)
        assert isinstance(result, dict)
        assert "transport_flux" in result
        assert isinstance(result["transport_flux"], torch.Tensor)
        # Flux should be negative (from high to low concentration)
        assert result["transport_flux"].item() < 0

    def test_physical_field_methods(self):
        """Test PhysicalFieldFunctor implements all PhysicalField abstract methods."""

        class TestFieldFunctor(PhysicalFieldFunctor):
            SPATIAL_DIMS = 2

            def forward(self, state, time, dt, **kwargs):
                return {'field_update': torch.zeros(10)}

            def get_dependencies(self):
                return []

            def get_value_at(self, coordinates):
                """Uniform test field of value 5.0 at every queried point."""
                return torch.full((coordinates.shape[0],), 5.0)

        functor = TestFieldFunctor(
            name='test_field',
            metadata=_test_metadata('test_field', ProcessCategory.PHYSICAL_FIELD),
        )

        # Test get_value_at
        coords = torch.randn(5, 2)  # 5 points in 2D
        values = functor.get_value_at(coords)
        assert isinstance(values, torch.Tensor)
        assert values.shape[0] == 5
        assert torch.allclose(values, torch.tensor(5.0))

        # Test compute_gradient (should return zeros in base implementation)
        gradient = functor.compute_gradient(coords)
        assert isinstance(gradient, torch.Tensor)

        # Test interpolate and update_field (should have default implementations)
        assert hasattr(functor, 'interpolate')
        assert hasattr(functor, 'update_field')

    def test_scale_coupling_methods(self):
        """Test ScaleCouplingFunctor implements upscale and downscale methods."""

        class TestScaleFunctor(ScaleCouplingFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {'scaled_data': torch.tensor(1.0)}

            def get_dependencies(self):
                return []

            def upscale(self, fine_data):
                """Override upscale with simple averaging."""
                if fine_data.dim() == 2:
                    return fine_data.mean(dim=-1)
                return fine_data.mean()

            def downscale(self, coarse_data):
                """Override downscale with simple replication."""
                if coarse_data.dim() == 1:
                    return coarse_data.unsqueeze(-1).repeat(1, 2)
                return coarse_data.repeat(2)

        functor = TestScaleFunctor(
            name='test_scale',
            metadata=_test_metadata('test_scale', ProcessCategory.SCALE_COUPLING),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        )

        # Test upscale
        fine_data = torch.randn(10, 20)
        coarse_data = functor.upscale(fine_data)
        assert isinstance(coarse_data, torch.Tensor)
        assert coarse_data.shape[0] == 10

        # Test downscale
        coarse_data = torch.randn(5)
        fine_data = functor.downscale(coarse_data)
        assert isinstance(fine_data, torch.Tensor)
        assert fine_data.numel() > coarse_data.numel()


class TestBridgeClassFunctionality(unittest.TestCase):
    """Test specific functionality provided by bridge classes."""

    def test_transport_flux_to_updates(self):
        """Test flux to update conversion in TransportMechanismFunctor."""

        class TestTransportFunctor(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {'concentration_flux': torch.tensor(0.1)}

            def get_dependencies(self):
                return []

        functor = TestTransportFunctor(
            name='test_transport',
            metadata=_test_metadata(
                'test_transport',
                ProcessCategory.TRANSPORT_MECHANISM,
            ),
        )

        # Test flux_to_updates method
        fluxes = {'concentration_flux': torch.tensor(0.1)}
        updates = functor.flux_to_updates(fluxes)

        assert isinstance(updates, dict)
        assert 'concentration_flux' in updates

    def test_transport_bridge_loads_canonical_variable_contracts(self):
        """Transport target units must come from the canonical contract directory."""

        class TestTransportFunctor(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_dependencies(self):
                return []

        functor = TestTransportFunctor(
            name="test_transport_contracts",
            metadata=_test_metadata(
                "test_transport_contracts",
                ProcessCategory.TRANSPORT_MECHANISM,
            ),
        )

        from relic.contracts.contract_loader import load_contract_registry

        canonical, state_units, _ = functor._declared_target_rate_units(
            "isoprene_emission", None
        )
        assert canonical == "isoprene_emission"
        registry = load_contract_registry()
        assert registry is load_contract_registry()
        assert state_units == registry.get_contract("isoprene_emission").udunits

    def test_scale_coupling_conservation_check(self):
        """Test conservation checking in ScaleCouplingFunctor."""

        class TestScaleFunctor(ScaleCouplingFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_dependencies(self):
                return []

        functor = TestScaleFunctor(
            name='test_scale',
            metadata=_test_metadata('test_scale', ProcessCategory.SCALE_COUPLING),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        )

        # Test conservation check
        input_data = torch.ones(10) * 2.0  # Total = 20
        output_conserved = torch.ones(5) * 4.0  # Total = 20 (conserved)
        output_violated = torch.ones(5) * 3.0  # Total = 15 (not conserved)

        is_conserved = functor.check_conservation_bridge(input_data, output_conserved)
        assert is_conserved

        is_violated = functor.check_conservation_bridge(input_data, output_violated)
        assert not is_violated


if __name__ == '__main__':
    # Run tests directly with Python
    import unittest
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    for test_class in [TestBridgeClassInheritance, TestBridgeClassInitialization,
                       TestAbstractMethodImplementation, TestBridgeClassFunctionality]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)

