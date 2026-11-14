"""
L1 tests for the canonical boundary-coupling parameter namespace.

The tests cover ParameterManager binding, physical-constant SSOT access, and
the exact boundary-layer conductance interface.

Test Level: L1 (Unit Verification)
Test Coverage: biosphere_atmosphere.py + coupling/biosphere_atmosphere.yaml
Author: RELIC Parameter System Refactoring Team
Date: 2025-10-17
"""

from pathlib import Path

class TestP1CouplingNamespaceAlignment:
    """Test P1.1: Coupling layer namespace alignment"""

    def test_biosphere_atmosphere_import(self):
        """The canonical boundary coupling is importable."""
        from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling

        assert BiosphereAtmosphereCoupling.__name__ == "BiosphereAtmosphereCoupling"

    def test_biosphere_atmosphere_no_auto_register(self):
        """Verify biosphere_atmosphere.py uses register_if_missing=False"""
        file_path = Path(__file__).parent.parent / "relic" / "coupling" / "internal" / "biosphere_atmosphere.py"
        content = file_path.read_text(encoding='utf-8')

        # Should NOT contain register_if_missing=True
        assert 'register_if_missing=True' not in content, \
            "biosphere_atmosphere.py still has register_if_missing=True"


    def test_coupling_parameter_yaml_exists(self):
        """The canonical boundary-coupling parameter file exists."""
        yaml_path = (
            Path(__file__).parent.parent
            / "relic"
            / "parameters"
            / "database"
            / "relic"
            / "coupling"
            / "biosphere_atmosphere.yaml"
        )
        assert yaml_path.exists(), "biosphere_atmosphere.yaml not found"

    def test_parameter_manager_canonical_boundary_parameters(self):
        """ParameterManager exposes the two canonical process parameters."""
        from relic.parameters import get_global_manager

        pm = get_global_manager()
        expected = {
            "coupling.scales.biosphere_atmosphere.leaf_characteristic_dimension": (
                "m",
                "float64",
            ),
            "coupling.scales.biosphere_atmosphere.boundary_layer_transfer_coefficient": (
                "m s-0.5",
                "float64",
            ),
        }

        for key, (units, dtype) in expected.items():
            spec = pm.get_parameter(key)
            assert spec is not None, f"Missing canonical parameter: {key}"
            assert spec.units == units
            assert spec.dtype == dtype
            assert spec.bounds is not None
            assert spec.description
            assert spec.references
            assert spec.tags
            assert spec.provenance

    def test_physical_constants_namespace(self):
        """Gas conversion constants are accessed only through the SSOT API."""
        from relic.utils.physical_constants import get_physical_constant

        assert float(get_physical_constant("standard_pressure")) == 101325.0
        assert float(get_physical_constant("gas_constant")) > 0.0

    def test_biosphere_atmosphere_instantiation(self):
        """Instantiation exposes only the canonical boundary-layer interface."""
        from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling

        functor = BiosphereAtmosphereCoupling(name="test_biosphere_atmosphere", config={})

        assert functor is not None
        assert functor.metadata.input_variables == [
            "friction_velocity",
            "air_temperature",
            "atmospheric_pressure",
        ]
        assert functor.metadata.output_variables == ["boundary_layer_conductance"]
        assert not hasattr(functor, "compute_stomatal_conductance")
        assert not hasattr(functor, "compute_penman_monteith_et")

    def test_namespace_consistency_check(self):
        """The removed namespace is absent and the canonical namespace is present."""
        from relic.parameters import get_global_manager

        pm = get_global_manager()

        assert pm.get_parameter(
            "relic.coupling.defaults.stomatal_conductance_min"
        ) is None
        assert pm.get_parameter(
            "coupling.scales.biosphere_atmosphere.leaf_characteristic_dimension"
        ) is not None

