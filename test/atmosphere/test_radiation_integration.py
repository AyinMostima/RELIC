"""
Integration test for RadiationField and RadiationFunctor coupling.

This test validates the integration between RadiationField (field representation)
and RadiationFunctor (physics calculations), ensuring proper data flow and
consistency within the radiation to energy balance chain.

References:
    - Campbell & Norman (1998). Environmental Biophysics
    - RELIC Architecture Guidelines
"""

from pathlib import Path

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.core.state import StateWriteMode, UpdateItem, UpdateMode
from relic.scientific.atmosphere.radiation import RadiationField, RadiationFunctor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RADIATION_FIELD_CERTIFICATE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "atmosphere_radiation_radiation_field.yaml"
)
RADIATION_FUNCTOR_CERTIFICATE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "atmosphere_radiation_radiation_functor.yaml"
)


def _update_value(outputs, key):
    """Return the tensor carried by one explicit runtime update."""
    update = outputs[key]
    assert isinstance(update, UpdateItem)
    assert isinstance(update.value, torch.Tensor)
    return update.value


def _radiation_state(**overrides):
    """Return the complete explicit RadiationFunctor state contract."""
    state = {
        'air_temperature': torch.tensor([298.15]),
        'incoming_shortwave_potential': torch.tensor([1000.0]),
        'cos_zenith': torch.tensor([0.8]),
        'surface_temperature': torch.tensor([298.15]),
        'relative_humidity': torch.tensor([0.6]),
        'surface_albedo': torch.tensor([0.2]),
        'surface_emissivity': torch.tensor([0.97]),
        'cloud_fraction': torch.tensor([0.0]),
    }
    state.update(overrides)
    return state


class TestRadiationIntegration:
    """Integration tests for RadiationField and RadiationFunctor."""

    @pytest.fixture
    def radiation_functor(self):
        """Create a RadiationFunctor for testing."""
        functor = RadiationFunctor(
            name="test_radiation_functor",
            device=torch.device("cpu")
        )
        return functor

    @pytest.fixture
    def radiation_field(self):
        """Create a RadiationField for testing."""
        field = RadiationField()
        return field

    def test_rejects_unimplemented_two_stream_proxy(self):
        """The former modified Beer-Lambert proxy is not a production model."""
        with pytest.raises(
            ValueError,
            match='Unsupported RadiationFunctor config keys: canopy',
        ):
            RadiationFunctor(
                name="two_stream_rejected",
                config={"canopy": {"model_type": "two_stream", "enabled": True}},
                device=torch.device("cpu"),
            )
        assert not hasattr(RadiationFunctor, '_two_stream_layers')

    def test_radiation_functor_output_consistent_with_field(self, radiation_functor):
        """Test that RadiationFunctor output can initialize RadiationField."""
        # Prepare state for RadiationFunctor
        state = _radiation_state()

        time = 0.0
        dt = 3600.0

        # Compute radiation fluxes
        fluxes = radiation_functor.forward(state, time, dt)

        # Verify functor produces expected outputs
        assert 'net_radiation' in fluxes
        assert 'shortwave_down' in fluxes
        assert 'longwave_down' in fluxes

        # All retained outputs carry explicit instantaneous replacement semantics.
        for key in (
            'net_radiation',
            'shortwave_down',
            'longwave_down',
            'ppfd',
        ):
            assert isinstance(fluxes[key], UpdateItem)
            assert fluxes[key].mode is UpdateMode.REPLACE
            assert fluxes[key].per_timestep is False
            assert torch.all(torch.isfinite(_update_value(fluxes, key)))

    def test_atmospheric_emissivity_is_an_explicit_derived_diagnostic(
        self,
        radiation_functor,
    ):
        """Keep algebraic emissivity out of the prognostic state transaction."""

        fluxes = radiation_functor.forward(
            _radiation_state(),
            time=0.0,
            dt=3600.0,
        )

        emissivity = fluxes["atmospheric_emissivity"]
        assert isinstance(emissivity, UpdateItem)
        assert emissivity.mode is UpdateMode.REPLACE
        assert emissivity.write_mode is StateWriteMode.DERIVED
        assert emissivity.per_timestep is False
        assert torch.all(torch.isfinite(emissivity.value))

    def test_field_forward_consumes_functor_output(self, radiation_field, radiation_functor):
        """Test that RadiationField can consume RadiationFunctor outputs."""
        # Step 1: Compute radiation with RadiationFunctor
        state = _radiation_state()

        time = 0.0
        dt = 3600.0

        # Get radiation fluxes
        fluxes = radiation_functor.forward(state, time, dt)
        net_radiation = _update_value(fluxes, 'net_radiation')

        # Step 2: Initialize field with grid
        grid_coords = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], dtype=torch.float32)

        # Use net_radiation to initialize field at all grid points
        initial_values = net_radiation.expand(4)  # Broadcast to 4 grid points
        radiation_field.initialize_field(grid_coords, initial_values)

        # Step 3: Call field forward to get field representation
        field_state = {
            'grid_coordinates': grid_coords,
            'radiation_field': initial_values,
        }
        field_output = radiation_field.forward(field_state, time, dt)

        # Verify field output
        assert 'radiation_field' in field_output
        assert _update_value(field_output, 'radiation_field').shape == (4,)
        assert torch.all(torch.isfinite(_update_value(field_output, 'radiation_field')))

    def test_complete_radiation_chain(self, radiation_field, radiation_functor):
        """Test complete radiation chain: Functor to Field to Downstream."""
        # Simulate a complete radiation calculation workflow

        # Step 1: Initial state
        state = _radiation_state()

        time = 0.0
        dt = 3600.0

        # Step 2: Compute radiation physics with RadiationFunctor
        radiation_fluxes = radiation_functor.forward(state, time, dt)

        # Step 3: Extract net radiation for field initialization
        net_radiation = _update_value(radiation_fluxes, 'net_radiation')

        # Step 4: Initialize spatial field
        grid_coords = torch.tensor([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0]
        ], dtype=torch.float32)

        field_values = net_radiation.expand(4)
        radiation_field.initialize_field(grid_coords, field_values)

        # Step 5: Get field representation at time t
        field_state = {
            'grid_coordinates': grid_coords,
            'radiation_field': field_values,
        }
        field_output = radiation_field.forward(field_state, time, dt)

        # Step 6: Query field at arbitrary points (e.g., for downstream coupling)
        query_points = torch.tensor([
            [5.0, 5.0],  # Center of grid
            [2.5, 2.5]   # Off-grid point
        ], dtype=torch.float32)

        interpolated_values = radiation_field.get_value_at(query_points)

        # Verify complete chain
        assert torch.all(
            torch.isfinite(_update_value(radiation_fluxes, 'net_radiation'))
        )
        assert torch.all(torch.isfinite(_update_value(field_output, 'radiation_field')))
        assert torch.all(torch.isfinite(interpolated_values))

        # Interpolated values should be within physical bounds
        assert torch.all(interpolated_values >= radiation_field.min_radiation)
        assert torch.all(interpolated_values <= radiation_field.max_radiation)

    def test_field_accepts_state_manager_snapshot(self, radiation_field):
        """Test a new field snapshot supplied by the external state transaction."""
        # Initialize field
        grid_coords = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0]
        ], dtype=torch.float32)

        initial_values = torch.tensor([200.0, 250.0], dtype=torch.float32)
        radiation_field.initialize_field(grid_coords, initial_values)

        time = 3600.0
        dt = 1800.0
        updated_values = torch.tensor([330.0, 380.0], dtype=torch.float32)
        field_output = radiation_field.forward(
            {
                'grid_coordinates': grid_coords,
                'radiation_field': updated_values,
            },
            time,
            dt,
        )

        assert 'radiation_field' in field_output
        field_value = _update_value(field_output, 'radiation_field')
        assert field_value.shape == (2,)
        assert torch.allclose(field_value, updated_values)
        assert torch.all(field_value >= radiation_field.min_radiation)
        assert torch.all(field_value <= radiation_field.max_radiation)

    def test_field_gradient_for_downstream_coupling(self, radiation_field):
        """Test gradient computation for coupling with transport processes."""
        # Initialize field with spatial variation
        grid_coords = torch.tensor([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0]
        ], dtype=torch.float32)

        # Create gradient: higher radiation in the east
        initial_values = torch.tensor([100.0, 200.0, 150.0, 250.0], dtype=torch.float32)
        radiation_field.initialize_field(grid_coords, initial_values)

        # Compute gradients at grid points
        state = {
            'grid_coordinates': grid_coords,
            'radiation_field': initial_values,
        }
        time = 0.0
        dt = 3600.0

        field_output = radiation_field.forward(
            state, time, dt,
            return_gradients=True
        )

        # Verify gradients are computed
        gradient_key = 'diagnostics.radiation_field.gradient'
        assert gradient_key in field_output
        gradient = _update_value(field_output, gradient_key)
        assert gradient.shape == (4, 2)  # [n_points, spatial_dims]
        assert torch.all(torch.isfinite(gradient))

    def test_multiple_state_snapshots(self, radiation_field):
        """Test multiple explicit state snapshots through the single field path."""
        # Initialize field
        grid_coords = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0]
        ], dtype=torch.float32)

        initial_values = torch.tensor([100.0, 100.0], dtype=torch.float32)
        radiation_field.initialize_field(grid_coords, initial_values)

        dt = 1800.0  # 30 minutes
        num_steps = 4
        field_snapshots = [
            torch.tensor([180.0, 180.0]),
            torch.tensor([520.0, 520.0]),
            torch.tensor([310.0, 310.0]),
            torch.tensor([80.0, 80.0]),
        ]

        field_history = []

        for i, field_values in enumerate(field_snapshots):
            time = i * dt
            field_output = radiation_field.forward(
                {
                    'grid_coordinates': grid_coords,
                    'radiation_field': field_values,
                },
                time,
                dt,
            )

            field_history.append(_update_value(field_output, 'radiation_field').clone())

        # Verify field evolution over time
        assert len(field_history) == num_steps
        for field_values in field_history:
            assert torch.all(torch.isfinite(field_values))
            assert torch.all(field_values >= radiation_field.min_radiation)
            assert torch.all(field_values <= radiation_field.max_radiation)

    def test_phase8_radiation_field_structure_hooks_are_declared(self, radiation_field):
        """Verify RadiationField exposes Phase 8 field structure hooks; its cochain
        layout is the initialized field (one grid point here)."""

        assert isinstance(radiation_field, IPHSDECFunctor)
        radiation_field.initialize_field(torch.tensor([[0.0, 0.0]]), torch.tensor([100.0]))
        forms = radiation_field.declared_forms()
        hodge_terms = radiation_field.build_hodge_terms()
        dissipative = radiation_field.dissipative_structure({})
        ports = radiation_field.ports()

        assert {form.semantic_name for form in forms} == {
            "radiation_flux_field",
            "radiation_flux_gradient",
        }
        assert (
            "atmosphere.radiation.field_bounds.generic_max_radiation"
            in radiation_field.required_parameters()
        )
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {"radiation_field_boundary_port"}

    def test_phase8_radiation_functor_is_an_honest_level2_boundary_wrapper(self, radiation_functor):
        """Reject pseudo-storage and pseudo-power claims for a flux-only process."""

        assert isinstance(radiation_functor, IPHSDECFunctor)
        forms = radiation_functor.declared_forms()
        hodge_terms = radiation_functor.build_hodge_terms()
        energy_terms = radiation_functor.energy_terms(
            {'net_radiation': torch.tensor(100.0)}
        )
        entropy_terms = radiation_functor.entropy_terms(
            {'net_radiation': torch.tensor(100.0)}
        )
        dissipative = radiation_functor.dissipative_structure({})
        ports = radiation_functor.ports()

        assert {form.semantic_name for form in forms} == {'radiative_boundary_flux'}
        assert (
            "atmosphere.radiation.atmospheric_transmittance"
            in radiation_functor.required_parameters()
        )
        certificate = load_iphs_equivalence_certificate(
            RADIATION_FUNCTOR_CERTIFICATE_PATH
        )
        assert set(radiation_functor.required_parameters()) == set(
            certificate.parameter_keys
        )
        assert not hasattr(radiation_functor, 'build_hodge_contract')
        assert hodge_terms == ()
        assert energy_terms == {}
        assert entropy_terms == {}
        assert dissipative is None
        assert ports == ()

    def test_phase8_radiation_certificates_declare_evidence_bounded_levels(self):
        """Verify field structure and flux-only wrapper use different levels."""

        for certificate_path, expected_level in (
            (RADIATION_FIELD_CERTIFICATE_PATH, 3),
            (RADIATION_FUNCTOR_CERTIFICATE_PATH, 2),
        ):
            certificate = load_iphs_equivalence_certificate(certificate_path)
            audit = certificate.validate()

            assert audit["passed"] is True
            assert certificate.canonical_role == "field"
            assert certificate.iphs_level == expected_level
            assert certificate.iphs_level_evidence[
                f'level_{expected_level}_limit'
            ]
            assert audit["runtime_evidence_status"] == "descriptive_only"


# Round 35: RadiationField declares one required_parameters binding set, equal to its
# metadata keys; the initialization default that no path used is gone, so a field is
# initialized only from supplied values; its projection bounds are named as bounds.
RADIATION_YAML_PATH = (
    PROJECT_ROOT / "relic" / "parameters" / "database" / "relic" / "atmosphere" / "radiation.yaml"
)


def test_radiation_field_layout_is_the_initialized_field():
    """Round 47: no literal layout stands in for an uninitialized field; the Hodge
    degrees of freedom are the field points."""
    import inspect

    parameter = inspect.signature(RadiationField.build_hodge_contract).parameters['n_dofs']
    assert parameter.default is inspect.Parameter.empty
    field = RadiationField()
    with pytest.raises(ValueError, match="initialize_field"):
        field.declared_forms()
    field.initialize_field(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), torch.tensor([100.0, 120.0, 140.0])
    )
    assert field.declared_forms()[0].n_dofs == 3
    (hodge,) = field.build_hodge_terms()
    assert tuple(hodge.matrix.shape) == (3, 3)
    assert tuple(field.dissipative_structure({})['M'].shape) == (3, 3)


def test_radiation_field_has_one_required_parameters_definition():
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(RadiationField)))
    methods = [node.name for node in tree.body[0].body if isinstance(node, ast.FunctionDef)]
    assert methods.count("required_parameters") == 1
    field = RadiationField()
    assert set(field.required_parameters()) == set(field.metadata.parameters)


def test_radiation_field_is_initialized_only_from_supplied_values():
    from relic.parameters import get_global_manager

    field = RadiationField()
    grid = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(TypeError):
        field.initialize_field(grid)
    manager = get_global_manager()
    assert manager.get_parameter(
        "atmosphere.radiation.field_defaults.canonical_default_radiation"
    ) is None


def test_radiation_field_bounds_are_named_as_bounds():
    import yaml

    from relic.parameters import get_global_manager

    field = RadiationField()
    manager = get_global_manager()
    for key in (
        "atmosphere.radiation.field_bounds.generic_min_radiation",
        "atmosphere.radiation.field_bounds.generic_max_radiation",
    ):
        assert key in field.required_parameters()
        assert manager.get_parameter(key) is not None
    entries = yaml.safe_load(RADIATION_YAML_PATH.read_text(encoding="utf-8-sig"))["parameters"]
    assert [key for key in entries if ".field_defaults." in key] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])




def test_radiation_field_bounds_have_one_source():
    """The projection bounds are the ParameterManager keys
    atmosphere.radiation.field_bounds.generic_{min,max}_radiation; the contract
    repeats no numeric bound."""
    from relic.contracts.contract_loader import load_contract_registry

    numerics = load_contract_registry().get_contract("radiation_field").raw["numerics"]
    assert numerics["lower_bound"] is None
    assert numerics["upper_bound"] is None
