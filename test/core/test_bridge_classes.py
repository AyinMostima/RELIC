"""
Test Suite for Functor Bridge Classes

This module tests the four bridge classes that unify the FunctorBase execution
shell with the four abstract base classes. These tests verify:

1. Proper multiple inheritance (both abstract class and FunctorBase)
2. Correct initialization with both scientific and system parameters
3. Default implementations of abstract methods
4. Proper MRO (Method Resolution Order) 
5. Parameter access through unified interface
6. Bridge-specific utility methods

References:
    - RELIC Architecture Document
    - Python Multiple Inheritance Documentation
"""

import dataclasses
import inspect
import typing

import pytest
import torch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import relic.scientific as scientific_package
from relic.core.dependency_graph import normalize_dependency_declaration
from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorBase,
    TransportMechanismFunctor, 
    PhysicalFieldFunctor,
    ScaleCouplingFunctor,
)
from relic.data.standards.unit_converter import UnitConverter
from relic.core.abstractions import (
    BiologicalProcess,
    TransportMechanism,
    PhysicalField,
    ScaleCoupling
)
from relic.core.functor import FunctorBase, ProcessCategory, FunctorMetadata
from relic.parameters import get_global_manager
from relic.parameters.ops import manager_registry
from relic.utils.physical_constants import get_physical_constant
from test.utils.forward_call import call_forward, call_transport_forward
from test.utils.functor_minimal_construction import (
    NOT_MINIMALLY_CONSTRUCTIBLE,
    construct_minimal,
    minimal_arguments,
    relic_functor_class_paths,
    resolve_class,
    verify_not_minimally_constructible,
)


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Each test runs with a clean, fully initialized parameter manager; the
    manager that was global before is put back afterwards."""
    get_global_manager()
    yield


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


class TestBiologicalProcessFunctor:
    """Test BiologicalProcessFunctor bridge class."""
    
    def test_multiple_inheritance(self):
        """Verify BiologicalProcessFunctor inherits from both base classes."""
        assert issubclass(BiologicalProcessFunctor, BiologicalProcess)
        assert issubclass(BiologicalProcessFunctor, FunctorBase)

    def test_bridge_has_no_state_environment_merge(self):
        """Drivers are state inputs; the bridge keeps no second merge path."""
        assert not hasattr(BiologicalProcessFunctor, "merge_state_environment")
    
    def test_initialization(self):
        """Test proper initialization of BiologicalProcessFunctor."""
        # Create concrete subclass for testing
        class TestBioProcess(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}
        
        functor = TestBioProcess(
            name="test_bio_process",
            metadata=_test_metadata("test_bio_process", ProcessCategory.BIOLOGICAL_PROCESS),
            process_name="photosynthesis",
        )
        
        # Check FunctorBase attributes
        assert functor.name == "test_bio_process"
        assert hasattr(functor, 'metadata')
        assert functor.metadata.category == ProcessCategory.BIOLOGICAL_PROCESS
        
        # Check BiologicalProcess attributes
        assert functor.process_name == "photosynthesis"
    
    def test_get_dependencies_exists(self):
        """Test that get_dependencies method exists."""
        class TestBioProcess(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}
        
        functor = TestBioProcess(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.BIOLOGICAL_PROCESS),
        )
        assert hasattr(functor, 'get_dependencies')
        deps = functor.get_dependencies()
        assert isinstance(deps, list)
        assert deps == []  # Default is empty
    
    def test_compute_rate_default(self):
        """Test forward-only biological entry implementation."""
        class TestBioFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {"test_rate": torch.tensor(1.0)}
        
        functor = TestBioFunctor(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.BIOLOGICAL_PROCESS),
        )
        state = {"substrate": torch.tensor(1.0), "temperature": torch.tensor(298.15)}

        result = call_forward(functor, state)
        assert isinstance(result, dict)
        assert "test_rate" in result
        assert isinstance(result["test_rate"], torch.Tensor)
    
    def test_enzyme_kinetics_is_not_a_process_parameter(self):
        """No algorithm reads a kinetic-family label: the contract and the bridge
        take none."""
        import inspect

        assert "enzyme_kinetics" not in inspect.signature(BiologicalProcess.__init__).parameters

        class TestBioFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

        with pytest.raises(TypeError, match="enzyme_kinetics"):
            TestBioFunctor(
                name="kinetics_probe",
                metadata=_test_metadata("kinetics_probe", ProcessCategory.BIOLOGICAL_PROCESS),
                enzyme_kinetics="hill",
            )

    def test_the_bridge_supplies_no_generic_temperature_law(self):
        """A concrete Functor owns its temperature law: the bridge defines no
        temperature_function and no q10 / arrhenius / optimal / linear law, so
        the abstract contract raises until the concrete class implements it."""
        for removed in (
            "temperature_function", "_temperature_response_q10", "_temperature_response_arrhenius",
            "_temperature_response_optimal", "_temperature_response_linear",
            "_resolve_activation_energy", "_resolve_temperature_sigma",
        ):
            assert removed not in BiologicalProcessFunctor.__dict__, removed

        class TestBioFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

        functor = TestBioFunctor(
            name="law_probe",
            metadata=_test_metadata("law_probe", ProcessCategory.BIOLOGICAL_PROCESS),
        )
        for generic in ("q10", "activation_energy", "temperature_sigma"):
            assert generic not in functor.metadata.parameters, generic
        with pytest.raises(RuntimeError, match="concrete BiologicalProcessFunctor"):
            functor.temperature_function(torch.tensor([298.15]))

        class OwnLawFunctor(TestBioFunctor):
            def temperature_function(self, temperature):
                return temperature / 298.15

        own = OwnLawFunctor(
            name="own_law_probe",
            metadata=_test_metadata("own_law_probe", ProcessCategory.BIOLOGICAL_PROCESS),
        )
        assert torch.allclose(own.temperature_function(torch.tensor([298.15])), torch.tensor([1.0]))

    def test_parameter_access(self):
        """Test parameter access through get_parameter_value."""
        class TestBioProcess(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}
        
        functor = TestBioProcess(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.BIOLOGICAL_PROCESS),
        )
        
        # Should have the proxy method
        assert hasattr(functor, 'get_parameter_value')
        
        # A missing parameter is None: no caller-supplied default exists.
        assert functor.get_parameter_value('nonexistent') is None


class TestTransportMechanismFunctor:
    """Test TransportMechanismFunctor bridge class."""

    def test_multiple_inheritance(self):
        """Verify TransportMechanismFunctor inherits from both base classes."""
        assert issubclass(TransportMechanismFunctor, TransportMechanism)
        assert issubclass(TransportMechanismFunctor, FunctorBase)

    @staticmethod
    def _declared_transport_class(label):
        class DeclaredTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = label

            def forward(self, state, time, dt, **kwargs):
                return {}

        return DeclaredTransport

    def test_transport_type_is_the_declared_class_attribute(self):
        functor = self._declared_transport_class("advection")(
            name="declared_transport",
            metadata=_test_metadata("declared_transport", ProcessCategory.TRANSPORT_MECHANISM),
        )
        assert functor.transport_type == "advection"
        assert functor.diffusivity is None

    def test_an_undeclared_transport_type_raises(self):
        class UndeclaredTransport(TransportMechanismFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

        with pytest.raises(TypeError, match="TRANSPORT_TYPE"):
            UndeclaredTransport(
                name="undeclared_transport",
                metadata=_test_metadata("undeclared_transport", ProcessCategory.TRANSPORT_MECHANISM),
            )

    def test_an_invalid_declared_label_raises(self):
        with pytest.raises(ValueError, match="Invalid transport type"):
            self._declared_transport_class("no_such_mechanism")(
                name="invalid_transport",
                metadata=_test_metadata("invalid_transport", ProcessCategory.TRANSPORT_MECHANISM),
            )

    @pytest.mark.parametrize("keyword, value", [("transport_type", "diffusion"), ("diffusivity", 1e-5)])
    def test_transport_constructor_keywords_are_rejected(self, keyword, value):
        with pytest.raises(TypeError, match=keyword):
            self._declared_transport_class("diffusion")(
                name="keyword_transport",
                metadata=_test_metadata("keyword_transport", ProcessCategory.TRANSPORT_MECHANISM),
                **{keyword: value},
            )
    
    def test_initialization(self):
        """Test proper initialization of TransportMechanismFunctor."""
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {}
        
        functor = TestTransport(
            name="test_transport",
            metadata=_test_metadata("test_transport", ProcessCategory.TRANSPORT_MECHANISM),
        )
        assert functor.diffusivity is None
        functor.set_scalar_diffusivity(1e-5)

        # Check FunctorBase attributes
        assert functor.name == "test_transport"
        assert hasattr(functor, 'metadata')
        assert functor.metadata.category == ProcessCategory.TRANSPORT_MECHANISM
        
        # Check TransportMechanism attributes  
        assert functor.transport_type == "diffusion"
        assert functor.diffusivity == 1e-5
    
    def test_get_dependencies_exists(self):
        """Test that get_dependencies method exists."""
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {}
        
        functor = TestTransport(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.TRANSPORT_MECHANISM),
        )
        assert hasattr(functor, 'get_dependencies')
        deps = functor.get_dependencies()
        assert isinstance(deps, list)
        assert deps == []
    
    def test_compute_flux_default(self):
        """Test default compute_flux implementation."""
        class TestTransportFunctor(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {"test_flux": torch.tensor(0.1)}
        
        functor = TestTransportFunctor(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.TRANSPORT_MECHANISM),
        )
        source = torch.tensor(1.0)
        sink = torch.tensor(0.5)
        
        fluxes = call_transport_forward(functor, source, sink)
        assert isinstance(fluxes, dict)
        assert "test_flux" in fluxes
        assert isinstance(fluxes["test_flux"], torch.Tensor)

    def test_generic_diffusivity_runtime_fails_fast(self):
        """The abstract bridge must not expose a second diffusivity law."""
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {}

        functor = TestTransport(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.TRANSPORT_MECHANISM),
        )
        functor.set_scalar_diffusivity(1e-5)

        with pytest.raises(RuntimeError, match="concrete transport"):
            functor.compute_diffusivity_tensor(
                torch.tensor([298.15]),
                torch.tensor([101325.0]),
            )

    def test_flux_to_updates(self):
        """Test flux_to_updates conversion."""
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {}
        
        functor = TestTransport(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.TRANSPORT_MECHANISM),
        )
        
        fluxes = {"water_flux": torch.tensor(0.1), "heat_flux": torch.tensor(5.0)}
        updates = functor.flux_to_updates(fluxes)
        
        assert updates == fluxes  # Default is identity
    
    def test_finalize_transport_updates(self):
        """Test finalize_transport_updates method."""
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {}
        
        functor = TestTransport(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.TRANSPORT_MECHANISM),
        )
        
        outputs = {
            "water_flux": torch.tensor(0.1),
            "diagnostics.transport.value": torch.tensor(42.0)
        }
        dt = torch.tensor(3600.0)
        # soil_water_storage declares kg m-2, so a flux declared in kg m-2 s-1
        # is already in the target rate and becomes F * dt without conversion.
        state = {"units": {"water_flux": "kg m-2 s-1"}}

        increments = functor.finalize_transport_updates(
            outputs,
            dt,
            state=state,
            mapping={"water_flux": "soil_water_storage"},
        )

        assert "soil_water_storage" in increments
        assert "diagnostics.transport.value" not in increments
        assert torch.allclose(increments["soil_water_storage"], torch.tensor(0.1 * 3600.0))

    @staticmethod
    def _plain_transport():
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {}

        return TestTransport(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.TRANSPORT_MECHANISM),
        )

    def test_transport_attributes_come_only_from_the_transport_contract(self):
        """No fabricated transport_type or diffusivity when TransportMechanism.__init__ was bypassed."""
        functor = self._plain_transport()
        for attribute in ("transport_type", "diffusivity"):
            saved = getattr(functor, attribute)
            delattr(functor, attribute)
            with pytest.raises(RuntimeError, match="TransportMechanism.__init__"):
                functor._init_transport_parameters()
            assert not hasattr(functor, attribute)
            setattr(functor, attribute, saved)

    def test_finalize_routes_a_declared_mapping_key_without_flux_in_its_name(self):
        """The mapping declares which outputs are fluxes and where they go."""
        functor = self._plain_transport()
        increments = functor.finalize_transport_updates(
            {"infiltration": torch.tensor(0.2)},
            torch.tensor(10.0),
            state={"units": {"infiltration": "kg m-2 s-1"}},
            mapping={"infiltration": "soil_water_storage"},
        )
        assert torch.allclose(increments["soil_water_storage"], torch.tensor(2.0))

    @pytest.mark.parametrize("mapping", [None, {}, {"other_flux": "soil_water_storage"}])
    def test_finalize_rejects_an_output_without_a_declared_target(self, mapping):
        """A flux-like name no longer derives a target; the key must be mapped."""
        functor = self._plain_transport()
        with pytest.raises(ValueError, match="no declared transport target"):
            functor.finalize_transport_updates(
                {"tracer_flux": torch.tensor(1.0)},
                1.0,
                state={"units": {"tracer_flux": "g C m-2 s-1"}},
                mapping=mapping,
            )

    def test_finalize_rejects_an_unmapped_non_diagnostic_output(self):
        functor = self._plain_transport()
        with pytest.raises(ValueError, match="diagnostic_value"):
            functor.finalize_transport_updates(
                {
                    "water_flux": torch.tensor(0.1),
                    "diagnostic_value": torch.tensor(42.0),
                },
                1.0,
                state={"units": {"water_flux": "kg m-2 s-1"}},
                mapping={"water_flux": "soil_water_storage"},
            )

    @pytest.mark.parametrize("drop_diagnostics", [True, False])
    def test_finalize_handles_the_diagnostics_namespace(self, drop_diagnostics):
        functor = self._plain_transport()
        increments = functor.finalize_transport_updates(
            {
                "water_flux": torch.tensor(0.1),
                "diagnostics.transport.courant": torch.tensor(0.4),
            },
            1.0,
            state={"units": {"water_flux": "kg m-2 s-1"}},
            mapping={"water_flux": "soil_water_storage"},
            drop_diagnostics=drop_diagnostics,
        )
        assert ("diagnostics.transport.courant" in increments) is (not drop_diagnostics)

    def test_finalize_commits_small_increments_exactly(self):
        """No silent mass removal: an increment F * dt below any tolerance is
        committed as computed (roundoff belongs to the conservation layers).

        Numeric example: F = 1e-14 kg m-2 s-1, dt = 60 s -> delta = 6e-13 kg m-2
        (the removed filter compared it with 1e-10 kg m-2 s-1 and returned 0).
        """
        functor = self._plain_transport()
        assert "boundary_flux_tolerance" not in functor.metadata.parameters
        increments = functor.finalize_transport_updates(
            {"water_flux": torch.tensor([1.0e-14, 2.0e-3], dtype=torch.float64)},
            torch.tensor(60.0, dtype=torch.float64),
            state={"units": {"water_flux": "kg m-2 s-1"}},
            mapping={"water_flux": "soil_water_storage"},
        )
        torch.testing.assert_close(
            increments["soil_water_storage"],
            torch.tensor([6.0e-13, 0.12], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_removed_flux_filter_leaves_no_orphan_tolerance_declaration(self):
        """No production code reads a boundary-flux filter tolerance, so neither
        the parameter database nor the hydrosphere catalog declares one."""
        from relic.scientific.hydrosphere.interface_mapping import HydrosphereInterfaceMapper

        manager = get_global_manager()
        assert manager.get("transport.numerical.boundary_flux_tolerance", required=False) is None
        assert "boundary_flux_tolerance" not in {
            name
            for names in HydrosphereInterfaceMapper().get_all_names().values()
            for name in names
        }

    def test_conservation_diagnostics_relative_change_uses_the_increment(self):
        """relative_change = |sum(F * dt)| / |sum(x)|; the increment already
        carries dt (F = 0.5 kg m-2 s-1, dt = 10 s, x = 100 kg m-2 -> 0.05)."""
        functor = self._plain_transport()
        increments = functor.finalize_transport_updates(
            {"water_flux": torch.tensor([0.5])},
            torch.tensor(10.0),
            state={
                "soil_water_storage": torch.tensor([100.0]),
                "units": {"water_flux": "kg m-2 s-1"},
            },
            mapping={"water_flux": "soil_water_storage"},
        )
        relative = increments["diagnostics.transport.relative_change_soil_water_storage"]
        torch.testing.assert_close(relative, torch.tensor(0.05))

    def test_conservation_diagnostics_errors_propagate(self):
        """A diagnostic failure on the governed path raises instead of being logged."""
        functor = self._plain_transport()
        with pytest.raises(RuntimeError):
            functor.finalize_transport_updates(
                {"water_flux": torch.tensor([0.5, 0.5])},
                torch.tensor(10.0),
                state={
                    "soil_water_storage": torch.tensor([100.0, 100.0, 100.0]),
                    "units": {"water_flux": "kg m-2 s-1"},
                },
                mapping={"water_flux": "soil_water_storage"},
            )

    def test_finalize_leaves_bounds_and_conservation_to_the_state_manager(self, caplog):
        """The bridge keeps no name-based bounds or conservation layer: a
        temperature beyond any hardcoded range is not judged here; bounds and
        budgets belong to StateManager and the documented conservation layers."""
        for removed in (
            "_enhanced_conservation_validation",
            "_is_conserved_variable",
            "_has_physical_bounds",
            "_get_physical_bounds",
        ):
            assert not hasattr(TransportMechanismFunctor, removed)
        functor = self._plain_transport()
        with caplog.at_level("WARNING"):
            increments = functor.finalize_transport_updates(
                {"heating_flux": torch.tensor([1.0])},
                60.0,
                state={
                    "temperature": torch.tensor([400.0]),
                    "units": {"heating_flux": "K s-1"},
                },
                mapping={"heating_flux": "temperature"},
            )
        assert torch.allclose(increments["temperature"], torch.tensor([60.0]))
        assert "bounds violation" not in caplog.text
        assert "net flux" not in caplog.text


class TestPhysicalFieldFunctor:
    """Test PhysicalFieldFunctor bridge class."""
    
    def test_multiple_inheritance(self):
        """Verify PhysicalFieldFunctor inherits from both base classes."""
        assert issubclass(PhysicalFieldFunctor, PhysicalField)
        assert issubclass(PhysicalFieldFunctor, FunctorBase)
    
    def test_initialization(self):
        """Test proper initialization of PhysicalFieldFunctor."""
        class TestField(PhysicalFieldFunctor):
            SPATIAL_DIMS = 3

            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_value_at(self, coordinates):
                return coordinates[..., 0]

        functor = TestField(
            name="temperature_field",
            metadata=_test_metadata("temperature_field", ProcessCategory.PHYSICAL_FIELD),
        )

        # Check FunctorBase attributes
        assert functor.name == "temperature_field"
        assert hasattr(functor, 'metadata')
        assert functor.metadata.category == ProcessCategory.PHYSICAL_FIELD

        # The one declared field property is the class attribute SPATIAL_DIMS;
        # identity and units come from metadata outputs and state contracts.
        assert functor.SPATIAL_DIMS == 3
        assert functor.validate_coordinates(torch.zeros(2, 3))
        for removed in ("field_name", "units", "default_value", "spatial_dims", "temporal_dependence"):
            assert not hasattr(functor, removed), removed
    
    def test_get_dependencies_exists(self):
        """Test that get_dependencies method exists."""
        class TestField(PhysicalFieldFunctor):
            SPATIAL_DIMS = None

            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_value_at(self, coordinates):
                raise NotImplementedError("point model (SPATIAL_DIMS = None): no coordinate query")
        
        functor = TestField(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.PHYSICAL_FIELD),
        )
        assert hasattr(functor, 'get_dependencies')
        deps = functor.get_dependencies()
        assert isinstance(deps, list)
        assert deps == []
    
    def test_abstract_methods_raise(self):
        """Test that abstract methods raise NotImplementedError."""
        class TestField(PhysicalFieldFunctor):
            SPATIAL_DIMS = 2

            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_value_at(self, coordinates):
                return coordinates[..., 0]

        functor = TestField(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.PHYSICAL_FIELD),
        )

        coords = torch.tensor([[0.0, 0.0]])
        gradient = functor.compute_gradient(coords)
        assert isinstance(gradient, torch.Tensor)
        assert gradient.shape == coords.shape
        assert torch.allclose(gradient, torch.zeros_like(coords))

        # The bridge keeps no evolve -> field_values -> flux default path.
        assert not hasattr(functor, "evolve")
    
    def test_update_field_to_flux(self):
        """Test update_field_to_flux conversion (update / dt under the named output)."""
        class TestField(PhysicalFieldFunctor):
            SPATIAL_DIMS = None

            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_value_at(self, coordinates):
                raise NotImplementedError("point model (SPATIAL_DIMS = None): no coordinate query")

        functor = TestField(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.PHYSICAL_FIELD),
        )

        field_update = torch.tensor([1.0, 2.0, 3.0])
        flux_dict = functor.update_field_to_flux(field_update, 2.0, "temperature")

        assert set(flux_dict) == {"temperature"}
        assert torch.equal(flux_dict["temperature"], field_update / 2.0)


class TestPhysicalFieldFunctorContract:
    """The bridge supplies no field identity, units, dimensionality, forward or
    get_value_at; a concrete class declares SPATIAL_DIMS."""

    @staticmethod
    def _field_class(dims=2, declare=True):
        namespace = {
            "forward": lambda self, state, time, dt, **kwargs: {},
            "get_value_at": lambda self, coordinates: coordinates[..., 0],
        }
        if declare:
            namespace["SPATIAL_DIMS"] = dims
        return type("ContractField", (PhysicalFieldFunctor,), namespace)

    def test_the_bridge_has_no_default_forward_or_value_query(self):
        for removed in ("forward", "get_value_at", "evolve", "_init_field_parameters"):
            assert removed not in PhysicalFieldFunctor.__dict__, removed

    def test_an_undeclared_spatial_dims_raises(self):
        with pytest.raises(TypeError, match="SPATIAL_DIMS"):
            self._field_class(declare=False)(
                name="undeclared_field",
                metadata=_test_metadata("undeclared_field", ProcessCategory.PHYSICAL_FIELD),
            )

    @pytest.mark.parametrize(
        "keyword, value",
        [("field_name", "t"), ("units", "K"), ("default_value", 0.0), ("spatial_dims", 2), ("temporal_dependence", True)],
    )
    def test_removed_field_keywords_are_rejected(self, keyword, value):
        with pytest.raises(TypeError, match=keyword):
            self._field_class()(
                name="keyword_field",
                metadata=_test_metadata("keyword_field", ProcessCategory.PHYSICAL_FIELD),
                **{keyword: value},
            )

    def test_update_field_to_flux_requires_dt_and_the_output_name(self):
        functor = self._field_class()(
            name="flux_field",
            metadata=_test_metadata("flux_field", ProcessCategory.PHYSICAL_FIELD),
        )
        update = torch.tensor([2.0, 4.0])
        assert torch.equal(functor.update_field_to_flux(update, 2.0, "test_output")["test_output"], update / 2.0)
        with pytest.raises(TypeError):
            functor.update_field_to_flux(update)


class TestScaleCouplingDeclaration:
    """The scale pair and the coupling type are declared by every coupling."""

    @pytest.mark.parametrize("missing", ["source_scale", "target_scale", "coupling_type"])
    def test_a_missing_scale_declaration_raises(self, missing):
        class Coupling(ScaleCouplingFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

        declared = {"source_scale": "leaf", "target_scale": "canopy", "coupling_type": "spatial"}
        del declared[missing]
        with pytest.raises(TypeError, match=missing):
            Coupling(
                name="scale_probe",
                metadata=_test_metadata("scale_probe", ProcessCategory.SCALE_COUPLING),
                **declared,
            )

    def test_the_scaling_contracts_take_their_method_without_default(self):
        import inspect

        from relic.core.abstractions import (
            HierarchicalScalingBase,
            ScaleCoupling,
            SpatialScalingBase,
            TemporalScalingBase,
        )

        required = {
            ScaleCoupling.__init__: ("coupling_type",),
            SpatialScalingBase.__init__: ("method", "preserve"),
            SpatialScalingBase.regrid: ("method", "preserve"),
            TemporalScalingBase.__init__: ("method", "pattern_type"),
            TemporalScalingBase.resample: ("method",),
            HierarchicalScalingBase.__init__: ("weight_type", "rule"),
        }
        for function, names in required.items():
            parameters = inspect.signature(function).parameters
            for name in names:
                assert parameters[name].default is inspect.Parameter.empty, (function.__qualname__, name)


class TestNamespaceDefaultMixinForwarding:
    """The mixin supplies the class namespace and source order and forwards every
    other lookup option unchanged; nothing is dropped silently."""

    @staticmethod
    def _probe():
        from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin

        class NamespaceProbe(NamespaceDefaultMixin, BiologicalProcessFunctor):
            parameter_lookup_namespace = "probe.namespace_default"
            parameter_lookup_source_order = ("config",)

            def forward(self, state, time, dt, **kwargs):
                return {}

        return NamespaceProbe(
            name="namespace_probe",
            config={"parameters": {"rate": 2.0}},
            metadata=_test_metadata("namespace_probe", ProcessCategory.BIOLOGICAL_PROCESS),
        )

    def test_required_is_forwarded(self):
        probe = self._probe()
        assert probe.get_parameter_value("absent_rate") is None
        with pytest.raises(ValueError, match="absent_rate"):
            probe.get_parameter_value("absent_rate", required=True)

    def test_bounds_and_validate_are_forwarded(self):
        probe = self._probe()
        assert float(probe.get_parameter_value("rate", required=True)) == 2.0
        with pytest.raises(ValueError, match="outside bounds"):
            probe.get_parameter_value("rate", bounds=(0.0, 1.0))
        assert float(probe.get_parameter_value("rate", bounds=(0.0, 1.0), validate=False)) == 2.0

    def test_expected_type_is_forwarded(self):
        probe = self._probe()
        with pytest.raises(ValueError, match="must be of type str"):
            probe.get_parameter_value("rate", expected_type=str)

    def test_no_drop_list_remains(self):
        import inspect

        import relic.core.functor.functor_bridge_utils as bridge_utils

        assert not hasattr(bridge_utils, "_DEFAULT_PARAMETER_KWARG_DROP_KEYS")
        assert not hasattr(bridge_utils.NamespaceDefaultMixin, "parameter_lookup_drop_keys")
        parameters = inspect.signature(
            bridge_utils.BridgeParameterMixin.normalize_parameter_lookup_kwargs
        ).parameters
        assert "drop_keys" not in parameters


class TestBridgeErrorPropagation:
    """Bridge helpers raise instead of substituting a default."""

    def test_require_scalar_trainability_lookup_failure_propagates(self, monkeypatch):
        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {}

        functor = TestTransport(
            name="probe",
            metadata=_test_metadata("probe", ProcessCategory.TRANSPORT_MECHANISM),
        )

        def _broken():
            raise RuntimeError("parameter database unavailable")

        monkeypatch.setattr(functor, "get_parameter_manager", _broken)
        with pytest.raises(RuntimeError, match="parameter database unavailable"):
            functor.require_scalar("any_rate", source_order=("config",))

    def test_conservation_ledger_lookup_failure_propagates(self, monkeypatch):
        import relic.core.functor.functor_bridge_utils as bridge_utils

        class _BrokenLedger:
            def get(self, key, **kwargs):
                raise ValueError(f"ledger entry {key} is malformed")

        monkeypatch.setattr(bridge_utils, "_get_parameter_manager", lambda: _BrokenLedger())
        metadata = _test_metadata("ledger_probe", ProcessCategory.TRANSPORT_MECHANISM)
        metadata.conservation_laws = ["mass"]

        class TestTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def forward(self, state, time, dt, **kwargs):
                return {}

        with pytest.raises(ValueError, match="malformed"):
            TestTransport(name="ledger_probe", metadata=metadata)

    def test_a_field_without_a_value_query_is_abstract(self):
        # The bridge has no default get_value_at that answers through forward;
        # a concrete field implements its coordinate query (or raises
        # NotImplementedError as a point model).
        class QuerylessField(PhysicalFieldFunctor):
            SPATIAL_DIMS = 1

            def forward(self, state, time, dt, **kwargs):
                raise ValueError("coordinates outside the field domain")

        with pytest.raises(TypeError, match="get_value_at"):
            QuerylessField(
                name="queryless_field",
                metadata=_test_metadata("queryless_field", ProcessCategory.PHYSICAL_FIELD),
            )

    def test_optional_namespaced_parameter_failure_is_not_absence(self, monkeypatch):
        class TestCoupling(ScaleCouplingFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

        functor = TestCoupling(
            name="namespaced_probe",
            metadata=_test_metadata("namespaced_probe", ProcessCategory.SCALE_COUPLING),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        )

        def _invalid(*_args, **_kwargs):
            raise ValueError("value outside bounds")

        monkeypatch.setattr(functor, "get_parameter_value", _invalid)
        with pytest.raises(ValueError, match="outside bounds"):
            functor.get_namespaced_parameter("any_key", "coupling", required=False)


def test_scale_coupling_attributes_come_only_from_the_scale_contract():
    """No fabricated source_scale / target_scale / coupling_type when ScaleCoupling.__init__ was bypassed."""

    class TestCoupling(ScaleCouplingFunctor):
        def forward(self, state, time, dt, **kwargs):
            return {}

    functor = TestCoupling(
        name="coupling_probe",
        metadata=_test_metadata("coupling_probe", ProcessCategory.SCALE_COUPLING),
        source_scale="fine",
        target_scale="coarse",
        coupling_type="spatial",
    )
    for attribute in ("source_scale", "target_scale", "coupling_type"):
        saved = getattr(functor, attribute)
        delattr(functor, attribute)
        with pytest.raises(RuntimeError, match="ScaleCoupling.__init__"):
            functor._init_coupling_parameters()
        assert not hasattr(functor, attribute)
        setattr(functor, attribute, saved)


def test_biological_process_carries_no_temperature_or_ph_label():
    """No algorithm read the temperature_response / ph_dependence labels: each
    concrete process evaluates its own temperature and pH laws. The contract
    takes neither label, the bridge rejects both as unexpected keywords, and the
    process metadata exports neither."""
    import inspect

    from relic.core.abstractions import BiologicalProcess

    parameters = inspect.signature(BiologicalProcess.__init__).parameters
    assert "temperature_response" not in parameters
    assert "ph_dependence" not in parameters

    class LabelProbe(BiologicalProcessFunctor):
        def forward(self, state, time, dt, **kwargs):
            return {}

    for label, value in (("temperature_response", "q10"), ("ph_dependence", True)):
        with pytest.raises(TypeError, match=label):
            LabelProbe(
                name="label_probe",
                metadata=_test_metadata("label_probe", ProcessCategory.BIOLOGICAL_PROCESS),
                **{label: value},
            )
    functor = LabelProbe(
        name="label_probe",
        metadata=_test_metadata("label_probe", ProcessCategory.BIOLOGICAL_PROCESS),
    )
    process_metadata = functor.get_process_metadata()
    for label in ("temperature_response", "ph_dependence"):
        assert label not in process_metadata
        assert label not in vars(functor)


class TestScaleCouplingFunctor:
    """Test ScaleCouplingFunctor bridge class."""
    
    def test_multiple_inheritance(self):
        """Verify ScaleCouplingFunctor inherits from both base classes."""
        assert issubclass(ScaleCouplingFunctor, ScaleCoupling)
        assert issubclass(ScaleCouplingFunctor, FunctorBase)
    
    def test_initialization(self):
        """Test proper initialization of ScaleCouplingFunctor."""
        class TestCoupling(ScaleCouplingFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}
        
        functor = TestCoupling(
            name="test_coupling",
            metadata=_test_metadata("test_coupling", ProcessCategory.SCALE_COUPLING),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial"
        )
        
        # Check FunctorBase attributes
        assert functor.name == "test_coupling"
        assert hasattr(functor, 'metadata')
        assert functor.metadata.category == ProcessCategory.SCALE_COUPLING
        
        # Check ScaleCoupling attributes
        assert functor.source_scale == "fine"
        assert functor.target_scale == "coarse"
        assert functor.coupling_type == "spatial"
    
    def test_get_dependencies_exists(self):
        """Test that get_dependencies method exists."""
        class TestCoupling(ScaleCouplingFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}
        
        functor = TestCoupling(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.SCALE_COUPLING),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        )
        assert hasattr(functor, 'get_dependencies')
        deps = functor.get_dependencies()
        assert isinstance(deps, list)
        assert deps == []
    
    def test_upscale_default(self):
        """Test default upscale implementation."""
        class TestScaleFunctor(ScaleCouplingFunctor):
            def forward(self, state, time, dt, **kwargs):
                if state.get('operation') == 'upscale':
                    return {"upscaled_data": torch.mean(state['fine_data'])}
                return {}
        
        functor = TestScaleFunctor(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.SCALE_COUPLING),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        )
        fine_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        coarse_data = functor.upscale(fine_data)
        assert isinstance(coarse_data, torch.Tensor)
        assert coarse_data == torch.mean(fine_data)
    
    def test_check_conservation_bridge(self):
        """Test conservation checking."""
        class TestCoupling(ScaleCouplingFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}
        
        functor = TestCoupling(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.SCALE_COUPLING),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        )
        
        input_data = torch.tensor([1.0, 2.0, 3.0])
        output_data = torch.tensor([2.0, 4.0])  # Same sum
        
        is_conserved = functor.check_conservation_bridge(input_data, output_data)
        assert is_conserved is True
        
        # Non-conserved case
        bad_output = torch.tensor([1.0, 1.0])
        is_conserved = functor.check_conservation_bridge(input_data, bad_output)
        assert is_conserved is False

    def test_check_conservation_bridge_missing_policy_fails_fast(self, monkeypatch):
        """Missing conservation policy parameters must remain blocking errors."""
        class TestCoupling(ScaleCouplingFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

        functor = TestCoupling(
            name="test",
            metadata=_test_metadata("test", ProcessCategory.SCALE_COUPLING),
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        )

        def _missing_parameter(*args, **kwargs):
            raise KeyError("missing coupling conservation policy")

        monkeypatch.setattr(
            FunctorBase,
            "get_parameter_value",
            _missing_parameter,
        )

        with pytest.raises(KeyError, match="missing coupling conservation policy"):
            functor.check_conservation_bridge(
                torch.tensor([1.0]),
                torch.tensor([1.0]),
            )


class TestUnitConverter:
    """Test UnitConverter utility class."""
    
    def test_soil_moisture_conversions(self):
        """Test soil moisture unit conversions."""
        # Volumetric to mass-based
        volume_fraction = torch.tensor(0.3)  # 30% volumetric water content
        soil_depth = 2.0  # 2 meters
        water_density = get_physical_constant(
            "relic.physical.constants.water.density_liquid"
        )
        
        mass_content = UnitConverter.soil_moisture_m3m3_to_kgm2(
            volume_fraction, soil_depth, water_density
        )
        
        expected = 0.3 * soil_depth * water_density
        assert torch.allclose(mass_content, torch.tensor(expected))
        
        # Mass-based to volumetric
        volumetric = UnitConverter.soil_moisture_kgm2_to_m3m3(
            mass_content, soil_depth, water_density
        )
        
        assert torch.allclose(volumetric, volume_fraction)


class TestMultipleInheritanceMRO:
    """Test Method Resolution Order (MRO) for all bridge classes."""
    
    def test_biological_process_mro(self):
        """Verify MRO for BiologicalProcessFunctor."""
        mro = BiologicalProcessFunctor.__mro__
        
        # Should be: BiologicalProcessFunctor -> BiologicalProcess -> FunctorBase -> ...
        assert mro[0] == BiologicalProcessFunctor
        assert BiologicalProcess in mro
        assert FunctorBase in mro
        
        # BiologicalProcess should come before FunctorBase
        bio_idx = mro.index(BiologicalProcess)
        functor_idx = mro.index(FunctorBase)
        assert bio_idx < functor_idx
    
    def test_all_bridge_classes_mro(self):
        """Verify all bridge classes have correct MRO."""
        bridge_classes = [
            (BiologicalProcessFunctor, BiologicalProcess),
            (TransportMechanismFunctor, TransportMechanism),
            (PhysicalFieldFunctor, PhysicalField),
            (ScaleCouplingFunctor, ScaleCoupling)
        ]
        
        for bridge_class, abstract_class in bridge_classes:
            mro = bridge_class.__mro__
            
            # Verify order
            assert mro[0] == bridge_class
            assert abstract_class in mro
            assert FunctorBase in mro
            
            # Abstract should come before FunctorBase
            abstract_idx = mro.index(abstract_class)
            functor_idx = mro.index(FunctorBase)
            assert abstract_idx < functor_idx


class TestConcreteFunctorImplementation:
    """Test that concrete functors can properly inherit from bridge classes."""
    
    def test_concrete_biological_functor(self):
        """Test a concrete implementation of BiologicalProcessFunctor."""
        
        class ConcretePhotosynethesis(BiologicalProcessFunctor):
            def __init__(self, **kwargs):
                # Extract q10_value before passing to parent
                self.q10_value = kwargs.pop('q10_value', 2.0)
                super().__init__(**kwargs)

            def temperature_function(self, temp):
                # The concrete Functor owns its law: Q10 = 2 about 298.15 K per 10 K.
                return 2.0 ** ((temp - 298.15) / 10.0)
                
            def forward(self, state, time, dt, **kwargs):
                # Simple photosynthesis model
                par = state.get('par', torch.tensor(1000.0))
                temp = state.get('temperature', torch.tensor(298.15))
                
                # Apply temperature response
                temp_factor = self.temperature_function(temp)
                
                # Simple light response
                gpp = 0.01 * par * temp_factor
                
                return {"gpp": gpp}
        
        # Create instance
        ps = ConcretePhotosynethesis(
            name="photosynthesis",
            metadata=_test_metadata("photosynthesis", ProcessCategory.BIOLOGICAL_PROCESS),
            q10_value=2.0,
            config={}  # Add empty config to satisfy FunctorBase
        )
        
        # Test forward
        state = {
            "par": torch.tensor(800.0),
            "temperature": torch.tensor(303.15)  # 30C
        }
        
        result = ps.forward(state, 0.0, 1.0)
        assert "gpp" in result
        assert result["gpp"] > 0
        
        # forward is the only public compute API
        assert callable(ps.forward)
        assert not hasattr(ps, "compute_rate")


_BRIDGE_CATEGORIES = [
    (BiologicalProcessFunctor, ProcessCategory.BIOLOGICAL_PROCESS),
    (TransportMechanismFunctor, ProcessCategory.TRANSPORT_MECHANISM),
    (PhysicalFieldFunctor, ProcessCategory.PHYSICAL_FIELD),
    (ScaleCouplingFunctor, ProcessCategory.SCALE_COUPLING),
]


def _point_model_value_query(self, coordinates):
    raise NotImplementedError("point model (SPATIAL_DIMS = None): no coordinate query")


def _bridge_subclass(bridge, static=None):
    """Build a concrete bridge subclass; ``static`` adds a class-level declaration."""
    namespace = {"forward": lambda self, state, time, dt, **kwargs: {}}
    if bridge is TransportMechanismFunctor:
        # A concrete transport class declares its descriptive label.
        namespace["TRANSPORT_TYPE"] = "diffusion"
    if bridge is PhysicalFieldFunctor:
        # A concrete field declares SPATIAL_DIMS and its coordinate query; the
        # probe is a point model.
        namespace["SPATIAL_DIMS"] = None
        namespace["get_value_at"] = _point_model_value_query
    if static is not None:
        namespace["get_dependencies"] = (
            lambda self: self._merge_declared_dependencies(static)
        )
    return type(f"Probe{bridge.__name__}", (bridge,), namespace)


def _bridge_instance(bridge, category, static=None, config=None):
    cls = _bridge_subclass(bridge, static)
    # A scale coupling declares the scale pair it couples and the coupling type.
    scales = (
        {"source_scale": "fine", "target_scale": "coarse", "coupling_type": "spatial"}
        if bridge is ScaleCouplingFunctor
        else {}
    )
    return cls(
        name="dependency_probe",
        config=config if config is not None else {},
        metadata=_test_metadata("dependency_probe", category),
        **scales,
    )


class TestDeclaredDependencyMerge:
    """One merge rule: static class declaration + instance-declared names.

    Instance-declared names come from ``config["dependencies"]``,
    ``set_dependencies`` and ``add_dependency``. They carry no category and
    are required same-step edges unless the static declaration already lists
    the name as optional or lagged.
    """

    @pytest.mark.parametrize("bridge,category", _BRIDGE_CATEGORIES)
    def test_bridge_default_returns_instance_declared_list(self, bridge, category):
        functor = _bridge_instance(bridge, category, config={"dependencies": ["cfg_dep"]})
        functor.add_dependency("probe_dep")
        functor.add_dependency("probe_dep")
        assert functor.get_dependencies() == ["cfg_dep", "probe_dep"]

    @pytest.mark.parametrize("bridge,category", _BRIDGE_CATEGORIES)
    def test_static_list_is_merged_with_instance_names(self, bridge, category):
        functor = _bridge_instance(bridge, category, static=["static_a", "static_b"])
        assert functor.get_dependencies() == ["static_a", "static_b"]
        functor.add_dependency("probe_dep")
        functor.add_dependency("static_a")
        assert functor.get_dependencies() == ["static_a", "static_b", "probe_dep"]
        functor.set_dependencies([])
        assert functor.get_dependencies() == ["static_a", "static_b"]

    def test_static_mapping_keeps_categories_and_adds_required(self):
        static = {"optional": ["opt_dep"], "lagged": ["lag_dep"]}
        functor = _bridge_instance(
            TransportMechanismFunctor,
            ProcessCategory.TRANSPORT_MECHANISM,
            static=static,
        )
        assert functor.get_dependencies() == static
        functor.add_dependency("probe_dep")
        functor.add_dependency("lag_dep")
        functor.add_dependency("opt_dep")
        merged = functor.get_dependencies()
        assert merged == {
            "optional": ["opt_dep"],
            "lagged": ["lag_dep"],
            "required": ["probe_dep"],
        }
        assert normalize_dependency_declaration(merged, owner="probe").canonical() == {
            "required": ["probe_dep"],
            "optional": ["opt_dep"],
            "lagged": ["lag_dep"],
        }
        # The static declaration object is never mutated by a merge.
        assert static == {"optional": ["opt_dep"], "lagged": ["lag_dep"]}

    def test_static_required_mapping_absorbs_instance_names(self):
        functor = _bridge_instance(
            PhysicalFieldFunctor,
            ProcessCategory.PHYSICAL_FIELD,
            static={"required": ["req_dep"], "optional": []},
        )
        functor.add_dependency("probe_dep")
        assert functor.get_dependencies() == {
            "required": ["req_dep", "probe_dep"],
            "optional": [],
        }

    @pytest.mark.parametrize(
        "static,error",
        [
            ({"mandatory": ["x"]}, ValueError),
            ({"required": "x"}, TypeError),
            (["x", ""], ValueError),
            (["x", 3], TypeError),
            ("x", TypeError),
        ],
    )
    def test_malformed_static_declaration_fails_fast(self, static, error):
        functor = _bridge_instance(
            ScaleCouplingFunctor,
            ProcessCategory.SCALE_COUPLING,
            static=static,
        )
        with pytest.raises(error):
            functor.get_dependencies()


def _contract_class_paths():
    """Every FUNCTOR_REGISTRY class plus every FunctorBase subclass defined in relic.

    The full scan fills the process-global FUNCTOR_REGISTRY that simulator
    bootstrap resolves default names through, so the registry is restored to
    its previous content before this collection-time helper returns.
    """
    registry = scientific_package.FUNCTOR_REGISTRY
    saved_registry = dict(registry)
    try:
        scientific_package.register_all_functors()
        registry_paths = set(registry.values())
    finally:
        registry.clear()
        registry.update(saved_registry)
    return sorted(registry_paths | set(relic_functor_class_paths()))


def test_contract_enumeration_leaves_the_functor_registry_unchanged():
    registry = scientific_package.FUNCTOR_REGISTRY
    saved_registry = dict(registry)
    try:
        enumerated = _contract_class_paths()

        assert registry == saved_registry
        assert set(saved_registry.values()) <= set(enumerated)
    finally:
        registry.clear()
        registry.update(saved_registry)


def _resolve_class(path):
    return resolve_class(path)


def _declares_domain_config(functor_cls):
    """True when the constructor declares ``config`` as a dataclass: a domain
    configuration object (for example an external adapter's CESMConfig), not
    the FunctorBase config mapping, so it carries no dependency declaration."""
    annotation = typing.get_type_hints(functor_cls.__init__).get("config")
    candidates = [
        arg for arg in typing.get_args(annotation) if arg is not type(None)
    ] or [annotation]
    return any(dataclasses.is_dataclass(arg) for arg in candidates)


def _merge_owned_by_one_owner(functor_cls):
    """The resolved get_dependencies is a bridge default or goes through the merge owner."""
    method = functor_cls.get_dependencies
    bridge_defaults = {
        bridge.get_dependencies
        for bridge in (
            BiologicalProcessFunctor,
            TransportMechanismFunctor,
            PhysicalFieldFunctor,
            ScaleCouplingFunctor,
        )
    }
    return method in bridge_defaults or (
        "_merge_declared_dependencies" in method.__code__.co_names
    )


class TestRegistryDependencyMergeContract:
    """Every Functor class in relic honours add_dependency in get_dependencies."""

    @pytest.fixture(autouse=True, scope="class")
    def _fresh_global_parameter_manager(self):
        """One fresh parameter manager for the whole class sweep.

        The probes only construct Functors and read dependency declarations,
        so a per-class reset keeps isolation from the other test classes
        without re-initializing the manager for each of the ~120 probes. The
        manager that was global before the class is put back afterwards.
        """
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(manager_registry, "_global_manager", None)
            get_global_manager()
            yield

    @pytest.mark.parametrize("class_path", _contract_class_paths())
    def test_registry_functor_merges_instance_declared_dependency(self, class_path):
        """add_dependency must reach get_dependencies for every Functor class.

        References:
        - docs/source/architecture_overview.md Section 4.2 (unified dependency
          declaration entry point) and Section 4.3 (dependency graph).

        Mathematical expression:
        - D_after = D_static U D_instance, where an instance name absent from
          D_static joins the required part; optional and lagged parts are kept.

        Parameter meanings:
        - D_static: class-level declaration normalized to required/optional/lagged.
        - D_instance: names declared through add_dependency / set_dependencies /
          config["dependencies"].

        Classes that cannot be constructed minimally are re-verified against
        their listed reason and must still resolve get_dependencies to a bridge
        default or to an override that calls the merge owner.
        """
        functor_cls = _resolve_class(class_path)
        if class_path in NOT_MINIMALLY_CONSTRUCTIBLE:
            assert verify_not_minimally_constructible(class_path)
            if isinstance(functor_cls, type) and issubclass(functor_cls, FunctorBase):
                assert _merge_owned_by_one_owner(functor_cls), class_path
            return

        assert _merge_owned_by_one_owner(functor_cls), class_path
        functor = construct_minimal(class_path)
        before = normalize_dependency_declaration(
            functor.get_dependencies(), owner=functor.name
        ).canonical()
        assert "probe_dep" not in before["required"] + before["optional"] + before["lagged"]

        functor.add_dependency("probe_dep")
        after = normalize_dependency_declaration(
            functor.get_dependencies(), owner=functor.name
        ).canonical()

        assert after["required"] == sorted(set(before["required"]) | {"probe_dep"})
        assert after["optional"] == before["optional"]
        assert after["lagged"] == before["lagged"]

    @pytest.mark.parametrize("class_path", _contract_class_paths())
    def test_constructor_forwards_config_declared_dependencies(self, class_path):
        """A constructor that accepts ``config`` must hand it to FunctorBase.

        ``config["dependencies"]`` is read only by FunctorBase; a constructor
        that keeps the config to itself (or overwrites ``self.config``) drops
        it, and one that rejects the key removes the declaration channel every
        other Functor has. The declared name must reach get_dependencies.
        Classes without a ``config`` parameter, or whose ``config`` is a domain
        object rather than the FunctorBase config mapping, have no such channel.
        """
        if class_path in NOT_MINIMALLY_CONSTRUCTIBLE:
            assert verify_not_minimally_constructible(class_path)
            return
        functor_cls = _resolve_class(class_path)
        if "config" not in inspect.signature(functor_cls.__init__).parameters:
            return
        if _declares_domain_config(functor_cls):
            return
        kwargs = minimal_arguments(class_path)
        base_config = kwargs.get("config")
        if base_config is not None and not isinstance(base_config, dict):
            return
        kwargs["config"] = {**(base_config or {}), "dependencies": ["config_dep"]}
        functor = functor_cls(**kwargs)
        declared = normalize_dependency_declaration(
            functor.get_dependencies(), owner=functor.name
        ).canonical()
        assert "config_dep" in (
            declared["required"] + declared["optional"] + declared["lagged"]
        ), class_path

    @pytest.mark.parametrize(
        "class_path,with_config,after_set_empty",
        [
            # Instance-level default providers: merged with config, rebound by an
            # explicit composition (set_dependencies / GraphExecutor links).
            (
                "relic.scientific.biosphere.photosynthesis.farquhar:FarquharPhotosynthesis",
                ["config_dep", "environmental_forcing", "atmospheric_radiation"],
                [],
            ),
            (
                "relic.scientific.biosphere.surrogates.simple_photosynthesis:SimpleCanopyPhotosynthesis",
                ["config_dep", "atmospheric_radiation"],
                [],
            ),
            # Class-level static declaration: always merged, never replaced.
            (
                "relic.scientific.biosphere.bvoc.synthesis.monoterpene:MonoterpeneSynthesis",
                ["farquhar_photosynthesis", "config_dep"],
                ["farquhar_photosynthesis"],
            ),
        ],
    )
    def test_constructor_keeps_config_declared_dependencies(
        self, class_path, with_config, after_set_empty
    ):
        """A constructor must not overwrite config["dependencies"] after FunctorBase loads it."""
        functor = _resolve_class(class_path)(config={"dependencies": ["config_dep"]})
        assert functor.get_dependencies() == with_config
        functor.set_dependencies([])
        assert functor.get_dependencies() == after_set_empty


def test_imports():
    """Test that all required classes can be imported."""
    from relic.core.functor import (
        BiologicalProcessFunctor,
        TransportMechanismFunctor,
        PhysicalFieldFunctor,
        ScaleCouplingFunctor,
    )
    from relic.data.standards.unit_converter import UnitConverter
    
    # All imports should succeed
    assert BiologicalProcessFunctor is not None
    assert TransportMechanismFunctor is not None
    assert PhysicalFieldFunctor is not None
    assert ScaleCouplingFunctor is not None
    assert UnitConverter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
