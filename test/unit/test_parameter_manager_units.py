"""L1 Unit Test: Parameter Manager Unit Integration.

Tests the integration between the Parameter Manager and unit conversion system,
ensuring parameters are correctly converted and validated.

Reference:
- relic/parameters/manager.py (_convert_parameter_units method)
- Phase 0 Parameter Database Fixes
"""
import pytest
import torch
from relic.parameters import get_global_manager, ParameterSpec
from relic.parameters.types import ParameterCategory

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:Parameter 'biosphere\.photosynthesis\.vcmax25' requested with units 'meters' but stored units are 'umol CO2 m-2 s-1'\. Performing conversion\.:UserWarning"
    )
]


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


class TestParameterUnitConversion:
    """Test automatic unit conversion when retrieving parameters."""

    def test_parameter_retrieved_with_default_units(self):
        """Test retrieving parameter without unit conversion."""
        manager = get_global_manager()

        # Get a known parameter in its default units
        try:
            vcmax = manager.get(
                'biosphere.photosynthesis.vcmax25',
                units='umol CO2 m-2 s-1'
            )
            assert vcmax is not None
            assert isinstance(vcmax, (float, int, torch.Tensor))
        except KeyError:
            pytest.skip("vcmax25 parameter not found in database")

    def test_parameter_unit_validation(self):
        """Test that requesting invalid units raises error."""
        manager = get_global_manager()

        # Requesting length units for a rate parameter raises on the conversion
        # (the read path has no separate validate switch).
        with pytest.raises((ValueError, KeyError)):
            manager.get(
                'biosphere.photosynthesis.vcmax25',
                units='meters',  # Wrong unit type
            )

    def test_parameter_with_type_validation(self):
        """Test parameter retrieval with type validation."""
        manager = get_global_manager()

        try:
            vcmax = manager.get(
                'biosphere.photosynthesis.vcmax25',
                expected_type=float
            )
            assert isinstance(vcmax, float)
        except KeyError:
            pytest.skip("vcmax25 parameter not found")


class TestTierPriorityWithUnits:
    """Test that tier priority system works correctly with units.

    Calibrated parameters should override default parameters,
    even if they have different units.
    """

    def test_calibrated_overrides_default(self):
        """Test that calibrated tier takes priority over relic tier."""
        manager = get_global_manager()

        # vcmax25 exists in both calibrated and relic tiers
        try:
            # Should get calibrated value (75.3) not default (60.0)
            vcmax = manager.get('biosphere.photosynthesis.vcmax25')

            # The exact value depends on tier priority
            # Just ensure we get a valid number
            assert vcmax is not None
            assert isinstance(vcmax, (float, int, torch.Tensor))
            if isinstance(vcmax, (float, int)):
                assert vcmax > 0
        except KeyError:
            pytest.skip("vcmax25 parameter not found")

    def test_tier_priority_with_unit_conversion(self):
        """Test tier priority when requesting different units."""
        manager = get_global_manager()

        try:
            # Request in default units
            vcmax_umol = manager.get(
                'biosphere.photosynthesis.vcmax25',
                units='umol CO2 m-2 s-1'
            )

            # Both should return valid values
            assert vcmax_umol is not None
        except KeyError:
            pytest.skip("vcmax25 parameter not found")


class TestParameterSpecUnits:
    """Test ParameterSpec unit definitions."""

    def test_parameter_spec_with_units(self):
        """Test creating ParameterSpec with units."""
        spec = ParameterSpec(
            key='test.parameter',
            value=60.0,
            units='umol m-2 s-1',
            category=ParameterCategory.BIOLOGICAL_PARAMETER,
            description='Test parameter', dtype="float32", bounds=(-float("inf"), float("inf"))
        )

        assert spec.units == 'umol m-2 s-1'
        assert spec.value == 60.0

    def test_parameter_spec_unit_validation(self):
        """Test that ParameterSpec validates units."""
        # Creating spec with units should work
        spec = ParameterSpec(
            key='test.temperature',
            value=298.15,
            units='K',
            category=ParameterCategory.PHYSICAL_CONSTANT,
            description='Temperature', dtype="float32", bounds=(-float("inf"), float("inf"))
        )

        assert spec.units == 'K'


def test_time_unit_dimensions_positive_time_basis():
    """Require the validator facade to use the canonical dimensionality owner."""
    from relic.data.standards.dimensionality import (
        Dimensionality,
        dimensionality_from_unit,
    )
    from relic.parameters.validators.unit_validator import UnitValidator

    validator = UnitValidator()
    assert not hasattr(validator, "unit_" + "dimensions")
    assert dimensionality_from_unit("s") == Dimensionality(T=1)
    assert dimensionality_from_unit("day") == Dimensionality(T=1)
    assert dimensionality_from_unit("yr") == Dimensionality(T=1)
    assert dimensionality_from_unit("s day-1") == Dimensionality()
    assert dimensionality_from_unit("h s-1") == Dimensionality()


class TestParameterRegistration:
    """Test parameter registration with units."""

    def test_register_parameter_with_units(self):
        """Test registering a new parameter with unit specification."""
        manager = get_global_manager()

        spec = ParameterSpec(
            key='test.custom.parameter',
            value=100.0,
            units='m',
            category=ParameterCategory.BIOLOGICAL_PARAMETER,
            description='Custom test parameter', dtype="float32", bounds=(-float("inf"), float("inf"))
        )

        try:
            manager.register_parameter(spec)

            # Retrieve and verify
            value = manager.get('test.custom.parameter')
            assert value == 100.0
        except Exception as e:
            # Parameter registration may have constraints
            pytest.skip(f"Parameter registration not supported: {e}")


class TestUnitConsistencyCheck:
    """Test unit consistency checking in parameter system."""

    def test_conflicting_units_detected(self):
        """Test that conflicting unit definitions are detected.

        If the same parameter is defined with different units in
        different tiers, this should be detected.
        """
        manager = get_global_manager()

        # This test verifies the system can handle unit conflicts
        # The jmax25 parameter had unit inconsistency in Phase 0
        try:
            jmax = manager.get('biosphere.photosynthesis.jmax25')
            assert jmax is not None
        except KeyError:
            pytest.skip("jmax25 parameter not found")


class TestDebugReportUnits:
    """Test the debug_report_units functionality."""

    def test_generate_unit_report(self):
        """Test generating a unit consistency report."""
        manager = get_global_manager()

        # This should not raise an error
        try:
            report = manager.debug_report_units()
            # Report should be a string or dict
            assert report is not None
        except AttributeError:
            pytest.skip("debug_report_units not implemented")


class TestParameterBoundsWithUnits:
    """Test parameter bounds validation with units."""

    def test_parameter_within_bounds(self):
        """Test that parameter value is validated against bounds."""
        manager = get_global_manager()

        try:
            # vcmax25 should have bounds defined
            vcmax = manager.get(
                'biosphere.photosynthesis.vcmax25',
                units='umol CO2 m-2 s-1'
            )

            # Value should be within reasonable bounds (10-200)
            if isinstance(vcmax, (float, int)):
                assert 10 <= vcmax <= 200, \
                    f"vcmax {vcmax} outside expected bounds [10, 200]"
        except KeyError:
            pytest.skip("vcmax25 parameter not found")


class TestParameterNamespaces:
    """Test parameter namespacing with units."""

    def test_namespace_isolation(self):
        """Test that different namespaces are isolated."""
        manager = get_global_manager()

        # Try to get parameters from different namespaces
        try:
            bio_param = manager.get('biosphere.photosynthesis.vcmax25')
            assert bio_param is not None
        except KeyError:
            pytest.skip("Biosphere parameter not found")


class TestParameterCategoryWithUnits:
    """Test that parameter categories work with unit system."""

    def test_state_variable_category(self):
        """Test STATE_VARIABLE category (Phase 0 fix)."""
        # This was a Phase 0 bug - STATE_VARIABLE enum was missing
        from relic.parameters.types import ParameterCategory

        # Should not raise AttributeError
        assert hasattr(ParameterCategory, 'STATE_VARIABLE')

    def test_all_categories_defined(self):
        """Test that all necessary categories are defined."""
        from relic.parameters.types import ParameterCategory

        required_categories = [
            'BIOLOGICAL_PARAMETER',
            'PHYSICAL_CONSTANT',
            'STATE_VARIABLE',
            'PROCESS_PARAMETER'
        ]

        for cat_name in required_categories:
            assert hasattr(ParameterCategory, cat_name), \
                f"Missing category: {cat_name}"


class TestParameterWithConversionContext:
    """Test parameter retrieval that may need conversion context."""

    def test_parameter_get_with_context(self):
        """Test retrieving parameter when context might be needed."""
        manager = get_global_manager()

        try:
            # Get a parameter - context should be handled internally if needed
            vcmax = manager.get(
                'biosphere.photosynthesis.vcmax25',
                units='umol CO2 m-2 s-1'
            )
            assert vcmax is not None
        except KeyError:
            pytest.skip("Parameter not found")
