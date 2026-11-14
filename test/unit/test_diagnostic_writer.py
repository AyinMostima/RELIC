"""
L1 Unit Tests for DiagnosticWriter

Tests the core functionality of DiagnosticWriter bridge layer for converting
ConservationSummary objects to CF-1.8 compliant xarray.Dataset.

Test Coverage:
- Initialization and configuration
- Single summary conversion
- Time-series batch conversion
- CF time coordinate generation
- Metadata registry operations
- CF-1.8 compliance validation
"""

import inspect
import sys
from pathlib import Path

import pytest
import torch
import xarray as xr
import numpy as np

# Add relic to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.data.io.diagnostic_writer import (
    DiagnosticWriter,
    DEFAULT_DIAGNOSTIC_METADATA,
    export_diagnostics_to_netcdf
)
from relic.core.state import ConservationSummary
import relic.utils.physical_constants as physical_constants_module
from relic.utils.time_constants import SECONDS_PER_DAY


@pytest.fixture
def sample_conservation_summary():
    """Create a sample ConservationSummary for testing."""
    from relic.core.state import ElementConservationReport
    
    summary = ConservationSummary(
        timestamp=3600.0,  # 1 hour
        total_violations=0,
        warnings=[],
        balance_error={
            'C': 0.001,
            'N': -0.0005,
            'P': 0.0002
        },
        tolerance_value=1e-4,
        dynamic_tolerance_used=False,
        dt=None,
        custom_checks={
            'photosynthesis.vcmax_limiting': 0.85,
            'photosynthesis.light_limiting': 0.65,
            'soil.water_stress': 0.45,
            'respiration.q10_factor': 2.1
        }
    )
    
    # Add per-element reports
    summary.carbon_report = ElementConservationReport(
        balance_error=0.001,
        relative_error=1e-5
    )
    summary.nitrogen_report = ElementConservationReport(
        balance_error=-0.0005,
        relative_error=5e-6
    )
    summary.phosphorus_report = ElementConservationReport(
        balance_error=0.0002,
        relative_error=2e-6
    )
    
    return summary


@pytest.fixture
def multiple_summaries():
    """Create multiple ConservationSummary objects for time-series testing."""
    from relic.core.state import ElementConservationReport
    
    summaries = []
    for i in range(5):
        timestamp = 3600.0 * (i + 1)  # Hourly
        
        summary = ConservationSummary(
            timestamp=timestamp,
            total_violations=0,
            warnings=[],
            balance_error={
                'C': 0.001 * (i + 1) * 0.1,
                'N': -0.0005 * (i + 1) * 0.1,
            },
            tolerance_value=1e-4,
            dynamic_tolerance_used=False,
            dt=None,
            custom_checks={
                'photosynthesis.vcmax_limiting': 0.8 + i * 0.02,
                'soil.water_stress': 0.4 + i * 0.05,
            }
        )
        
        # Add per-element reports
        summary.carbon_report = ElementConservationReport(
            balance_error=0.001 * (i + 1) * 0.1,
            relative_error=1e-5 * (i + 1)
        )
        summary.nitrogen_report = ElementConservationReport(
            balance_error=-0.0005 * (i + 1) * 0.1,
            relative_error=5e-6 * (i + 1)
        )
        
        summaries.append(summary)
    
    return summaries


class TestDiagnosticWriterInitialization:
    """Test DiagnosticWriter initialization and configuration."""

    def test_default_initialization(self):
        """Test initialization with default metadata registry."""
        writer = DiagnosticWriter()
        
        assert writer.metadata_registry is not None
        assert len(writer.metadata_registry) > 0
        assert 'carbon_balance_error' in writer.metadata_registry
        assert writer.cf_conventions is not None

    def test_custom_metadata_initialization(self):
        """Test initialization with custom metadata registry."""
        custom_metadata = {
            'custom_diagnostic': {
                'standard_name': 'custom_diagnostic_variable',
                'units': 'kg m-2',
                'long_name': 'Custom Diagnostic Variable'
            }
        }
        
        writer = DiagnosticWriter(metadata_registry=custom_metadata)
        
        assert 'custom_diagnostic' in writer.metadata_registry
        assert writer.metadata_registry['custom_diagnostic']['units'] == 'kg m-2'

    def test_explicit_empty_metadata_registry_stays_empty(self):
        assert DiagnosticWriter(metadata_registry={}).metadata_registry == {}

    def test_default_metadata_registry_completeness(self):
        """Test that DEFAULT_DIAGNOSTIC_METADATA contains required entries."""
        required_diagnostics = [
            'carbon_balance_error',
            'nitrogen_balance_error',
            'phosphorus_balance_error',
            'water_balance_error',
            'dynamic_tolerance',
            'conservation_violations'  # Note: not 'violation_count'
        ]
        
        for diag in required_diagnostics:
            assert diag in DEFAULT_DIAGNOSTIC_METADATA, f"Missing required diagnostic: {diag}"
            assert 'standard_name' not in DEFAULT_DIAGNOSTIC_METADATA[diag]
            if diag == 'dynamic_tolerance':
                assert 'units' not in DEFAULT_DIAGNOSTIC_METADATA[diag]
                assert 'ledger' in DEFAULT_DIAGNOSTIC_METADATA[diag]['comment']
            else:
                assert 'units' in DEFAULT_DIAGNOSTIC_METADATA[diag]
            assert 'long_name' in DEFAULT_DIAGNOSTIC_METADATA[diag]


class TestConservationSummaryConversion:
    """Test single ConservationSummary conversion to xarray.Dataset."""

    def test_convert_single_summary(self, sample_conservation_summary):
        """Test converting a single ConservationSummary to Dataset."""
        writer = DiagnosticWriter()
        
        dataset = writer.convert_conservation_summary(
            summary=sample_conservation_summary,
            time_value=3600.0,
            reference_date="1850-01-01"
        )
        
        # Check dataset structure
        assert isinstance(dataset, xr.Dataset)
        assert 'time' in dataset.coords
        
        # Check balance error variables
        assert 'carbon_balance_error' in dataset.data_vars
        assert 'nitrogen_balance_error' in dataset.data_vars
        assert 'phosphorus_balance_error' in dataset.data_vars
        
        # Check relative error variables
        assert 'carbon_relative_error' in dataset.data_vars
        assert 'nitrogen_relative_error' in dataset.data_vars
        
        # Check custom diagnostics (note: dots replaced with underscores for NetCDF)
        assert 'photosynthesis_vcmax_limiting' in dataset.data_vars
        assert 'soil_water_stress' in dataset.data_vars

    def test_local_diagnostic_metadata_and_cf_time(self, sample_conservation_summary):
        """Local diagnostics must not invent official CF standard names."""
        writer = DiagnosticWriter()
        
        dataset = writer.convert_conservation_summary(
            summary=sample_conservation_summary,
            time_value=3600.0
        )
        
        # Check carbon_balance_error metadata
        carbon_var = dataset['carbon_balance_error']
        assert 'standard_name' not in carbon_var.attrs
        assert hasattr(carbon_var, 'units')
        assert hasattr(carbon_var, 'long_name')
        assert carbon_var.attrs['units'] == 'g m-2'
        assert carbon_var.attrs['element'] == 'C'
        
        # Check time coordinate metadata
        time_var = dataset['time']
        assert hasattr(time_var, 'units')
        assert hasattr(time_var, 'calendar')
        assert 'days since' in time_var.attrs['units']

    def test_default_diagnostics_pass_canonical_metadata_subset(self):
        from relic.data.standards.cf_conventions import CFConventions

        for name, metadata in DEFAULT_DIAGNOSTIC_METADATA.items():
            result = CFConventions.validate_cf_compliance(metadata)
            assert result.is_valid, (name, result.issues)
            assert result.membership_checked is False

    def test_unknown_diagnostic_does_not_infer_units_or_cf_identity(self):
        writer = DiagnosticWriter()
        summary = ConservationSummary(custom_checks={"unknown.energy_metric": 3.0}, total_violations=0, tolerance_value=1e-6)
        dataset = writer.convert_conservation_summary(summary, time_value=1.0)

        metric = dataset["unknown_energy_metric"]
        assert metric.item() == 3.0
        assert "standard_name" not in metric.attrs
        assert "units" not in metric.attrs
        assert "unspecified" in metric.attrs["comment"]

    def test_time_coordinate_conversion(self, sample_conservation_summary):
        """Test time coordinate conversion from simulation seconds to CF days."""
        writer = DiagnosticWriter()
        
        # Test conversion: 3600s = 1 hour = 1/24 days
        dataset = writer.convert_conservation_summary(
            summary=sample_conservation_summary,
            time_value=3600.0,
            reference_date="2000-01-01"
        )
        
        time_value = dataset['time'].item()
        expected_value = 3600.0 / 86400.0  # Convert seconds to days
        
        assert np.isclose(time_value, expected_value, rtol=1e-6)

    def test_element_balance_values(self, sample_conservation_summary):
        """Test that element balance values are correctly transferred."""
        writer = DiagnosticWriter()
        
        dataset = writer.convert_conservation_summary(
            summary=sample_conservation_summary,
            time_value=3600.0
        )
        
        # Check carbon balance
        carbon_balance = dataset['carbon_balance_error'].item()
        assert np.isclose(carbon_balance, 0.001, rtol=1e-6)
        
        # Check nitrogen balance
        nitrogen_balance = dataset['nitrogen_balance_error'].item()
        assert np.isclose(nitrogen_balance, -0.0005, rtol=1e-6)

    def test_custom_diagnostics_preservation(self, sample_conservation_summary):
        """Test that custom diagnostics are preserved in Dataset."""
        writer = DiagnosticWriter()
        
        dataset = writer.convert_conservation_summary(
            summary=sample_conservation_summary,
            time_value=3600.0
        )
        
        # Check photosynthesis diagnostics (dots replaced with underscores)
        vcmax_limiting = dataset['photosynthesis_vcmax_limiting'].item()
        assert np.isclose(vcmax_limiting, 0.85, rtol=1e-6)
        
        # Check soil diagnostics (dots replaced with underscores)
        water_stress = dataset['soil_water_stress'].item()
        assert np.isclose(water_stress, 0.45, rtol=1e-6)


class TestTimeSeriesConversion:
    """Test batch conversion of multiple ConservationSummary objects."""

    def test_convert_multiple_summaries(self, multiple_summaries):
        """Test converting multiple summaries to time-series Dataset."""
        writer = DiagnosticWriter()
        
        time_coords = [summary.timestamp for summary in multiple_summaries]
        
        dataset = writer.convert_summaries_to_dataset(
            summaries=multiple_summaries,
            time_coords=time_coords,
            reference_date="1850-01-01"
        )
        
        # Check time dimension
        assert 'time' in dataset.dims
        assert dataset.sizes['time'] == 5
        
        # Check variable shapes
        assert dataset['carbon_balance_error'].shape == (5,)
        assert dataset['nitrogen_balance_error'].shape == (5,)

    def test_time_series_monotonicity(self, multiple_summaries):
        """Test that time coordinate is monotonically increasing."""
        writer = DiagnosticWriter()
        
        time_coords = [summary.timestamp for summary in multiple_summaries]
        
        dataset = writer.convert_summaries_to_dataset(
            summaries=multiple_summaries,
            time_coords=time_coords
        )
        
        time_values = dataset['time'].values
        
        # Check monotonicity
        assert np.all(np.diff(time_values) > 0)

    def test_time_series_data_integrity(self, multiple_summaries):
        """Test data integrity across time series."""
        writer = DiagnosticWriter()
        
        time_coords = [summary.timestamp for summary in multiple_summaries]
        
        dataset = writer.convert_summaries_to_dataset(
            summaries=multiple_summaries,
            time_coords=time_coords
        )
        
        # Check first timestep values
        carbon_balance_0 = float(dataset['carbon_balance_error'].isel(time=0).values)
        expected_0 = 0.001 * 1 * 0.1
        assert np.isclose(carbon_balance_0, expected_0, rtol=1e-5)
        
        # Check last timestep values
        carbon_balance_4 = float(dataset['carbon_balance_error'].isel(time=4).values)
        expected_4 = 0.001 * 5 * 0.1
        assert np.isclose(carbon_balance_4, expected_4, rtol=1e-5)

    def test_empty_summaries_handling(self):
        """Test handling of empty summaries list."""
        writer = DiagnosticWriter()
        
        dataset = writer.convert_summaries_to_dataset(
            summaries=[],
            time_coords=[],
            reference_date="1850-01-01"
        )
        
        # Should return empty Dataset
        assert isinstance(dataset, xr.Dataset)
        assert len(dataset.data_vars) == 0


class TestCFTimeCoordinate:
    """Test CF-compliant time coordinate generation."""

    def test_time_coordinate_creation(self):
        """Test basic CF time coordinate creation."""
        writer = DiagnosticWriter()
        
        time_values = [0.0, 3600.0, 7200.0, 10800.0]  # 0, 1h, 2h, 3h
        
        time_coord = writer.create_cf_time_coordinate(
            time_values=time_values,
            reference_date="2000-01-01",
            calendar="gregorian"
        )
        
        assert isinstance(time_coord, xr.DataArray)
        assert len(time_coord) == 4
        assert time_coord.attrs['units'] == 'days since 2000-01-01'
        assert time_coord.attrs['calendar'] == 'gregorian'

    def test_time_coordinate_conversion_formula(self):
        """Test time coordinate conversion formula: t_cf = t_sim / 86400.0."""
        writer = DiagnosticWriter()
        
        time_values = [86400.0, 2 * 86400.0, 3 * 86400.0]  # 1, 2, 3 days
        
        time_coord = writer.create_cf_time_coordinate(
            time_values=time_values,
            reference_date="2000-01-01"
        )
        
        # Check conversion: seconds / 86400 = days
        assert np.isclose(float(time_coord[0].values), 1.0, rtol=1e-6)
        assert np.isclose(float(time_coord[1].values), 2.0, rtol=1e-6)
        assert np.isclose(float(time_coord[2].values), 3.0, rtol=1e-6)

    def test_time_coordinate_uses_seconds_per_day_ssot(self, monkeypatch):
        writer = DiagnosticWriter()
        monkeypatch.setattr(
            physical_constants_module,
            "get_physical_constant",
            lambda name: float(SECONDS_PER_DAY) / 2.0,
        )

        time_coord = writer.create_cf_time_coordinate(
            [float(SECONDS_PER_DAY)],
            reference_date="2000-01-01",
        )

        assert float(time_coord[0].values) == pytest.approx(2.0)

    def test_time_coordinate_has_no_local_day_literal(self):
        source = inspect.getsource(DiagnosticWriter.create_cf_time_coordinate)

        assert "86400" not in source
        assert "get_physical_constant" in source

    def test_different_calendars(self):
        """Test support for different CF calendar types."""
        writer = DiagnosticWriter()
        
        time_values = [0.0, 86400.0]
        
        for calendar in ['gregorian', 'noleap', '360_day']:
            time_coord = writer.create_cf_time_coordinate(
                time_values=time_values,
                reference_date="1850-01-01",
                calendar=calendar
            )
            
            assert time_coord.attrs['calendar'] == calendar

    def test_time_coordinate_metadata(self):
        """Test required CF time coordinate metadata."""
        writer = DiagnosticWriter()
        
        time_coord = writer.create_cf_time_coordinate(
            time_values=[0.0, 3600.0],
            reference_date="2000-01-01"
        )
        
        # Check required attributes
        assert 'units' in time_coord.attrs
        assert 'calendar' in time_coord.attrs
        assert 'standard_name' in time_coord.attrs
        assert time_coord.attrs['standard_name'] == 'time'


class TestMetadataRegistry:
    """Test diagnostic metadata registry operations."""

    def test_register_custom_diagnostic(self):
        """Test registering custom diagnostic metadata."""
        writer = DiagnosticWriter()
        
        initial_count = len(writer.metadata_registry)
        
        writer.register_diagnostic_metadata(
            diagnostic_key='custom_flux',
            standard_name='custom_carbon_flux',
            units='kg C m-2 s-1',
            long_name='Custom Carbon Flux',
            valid_range=(-1e-3, 1e-3),
            comment='Custom diagnostic for testing'
        )
        
        assert len(writer.metadata_registry) == initial_count + 1
        assert 'custom_flux' in writer.metadata_registry
        
        metadata = writer.metadata_registry['custom_flux']
        assert metadata['standard_name'] == 'custom_carbon_flux'
        assert metadata['units'] == 'kg C m-2 s-1'
        assert metadata['valid_range'] == (-1e-3, 1e-3)

    def test_override_existing_metadata(self):
        """Test that registering existing key overrides metadata."""
        writer = DiagnosticWriter()
        
        # Register initial metadata
        writer.register_diagnostic_metadata(
            diagnostic_key='test_var',
            standard_name='test_variable_v1',
            units='kg',
            long_name='Test Variable Version 1'
        )
        
        # Override with new metadata
        writer.register_diagnostic_metadata(
            diagnostic_key='test_var',
            standard_name='test_variable_v2',
            units='g',
            long_name='Test Variable Version 2'
        )
        
        metadata = writer.metadata_registry['test_var']
        assert metadata['standard_name'] == 'test_variable_v2'
        assert metadata['units'] == 'g'

    def test_metadata_used_in_conversion(self):
        """Test that registered metadata is used during conversion."""
        writer = DiagnosticWriter()
        
        # Register custom metadata
        writer.register_diagnostic_metadata(
            diagnostic_key='custom_check',
            standard_name='custom_conservation_check',
            units='dimensionless',
            long_name='Custom Conservation Check'
        )
        
        # Create summary with custom diagnostic
        from relic.core.state import ElementConservationReport
        
        summary = ConservationSummary(
            timestamp=3600.0,
            total_violations=0,
            warnings=[],
            balance_error={'C': 0.0},
            tolerance_value=1e-4,
            dynamic_tolerance_used=False,
            dt=None,
            custom_checks={'custom_check': 0.95}
        )
        
        summary.carbon_report = ElementConservationReport(
            balance_error=0.0,
            relative_error=1e-6
        )
        
        dataset = writer.convert_conservation_summary(
            summary=summary,
            time_value=3600.0
        )
        
        # Check that custom metadata was applied
        assert 'custom_check' in dataset.data_vars
        custom_var = dataset['custom_check']
        assert custom_var.attrs['standard_name'] == 'custom_conservation_check'
        assert custom_var.attrs['units'] == 'dimensionless'


class TestConvenienceFunction:
    """Test export_diagnostics_to_netcdf convenience function."""

    @pytest.mark.parametrize("element,symbol,cf_unit", [
        ("carbon", "C", "g m-2"), ("nitrogen", "N", "g m-2"),
        ("phosphorus", "P", "g m-2"), ("water", "W", "kg m-2"),
    ])
    def test_real_budget_residual_keeps_owner_units_and_is_not_masked(self, tmp_path, element, symbol, cf_unit):
        import netCDF4
        from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode
        from relic.data.standards.unit_converter import unit_converter

        unit = unit_converter.get_element_budget_units(element)["storage"]
        manager = StateManager(schema={
            "budget_pool": StateVariableSpec("budget_pool", unit, conserved=True, required=False),
        }, autoload_contract_registry=False, device="cpu")
        manager.set_conservation_ledger({element: ["budget_pool"]})
        manager.set_state({"budget_pool": torch.tensor(2.0, dtype=torch.float64)}, validate=False)
        summary = manager.generate_conservation_report({
            "budget_pool": UpdateItem(
                torch.tensor(1.0, dtype=torch.float64), mode=UpdateMode.ACCUMULATE,
                unit=unit, per_timestep=True, element=symbol,
            ),
        }, current_time=86400.0, dt=86400.0, store_history=False)
        assert summary.balance_error[symbol] == pytest.approx(1.0)
        path = tmp_path / (element + ".nc")
        result = export_diagnostics_to_netcdf([summary], [summary.timestamp], path, delivery_key=element)
        assert result.validation_passed
        with netCDF4.Dataset(path) as dataset:
            error = dataset[element + "_balance_error"]
            assert error.units == cf_unit
            assert not np.ma.is_masked(error[:])
            assert float(error[0]) == pytest.approx(1.0)
            if element != "water":
                relative = dataset[element + "_relative_error"]
                assert not np.ma.is_masked(relative[:])
                assert float(relative[0]) == pytest.approx(0.5)
            tolerance = dataset["dynamic_tolerance"]
            assert not np.ma.is_masked(tolerance[:])
            assert float(tolerance[0]) == pytest.approx(summary.tolerance_value)
            assert "units" not in tolerance.ncattrs()

    def test_budget_unit_label_cannot_reinterpret_canonical_values(self, tmp_path):
        with pytest.raises(ValueError, match="budget.*unit|unit.*budget"):
            export_diagnostics_to_netcdf(
                [ConservationSummary(timestamp=1.0, balance_error={"C": 1.0}, total_violations=0, tolerance_value=1e-6)], [1.0],
                tmp_path / "wrong.nc", metadata_registry={"carbon_balance_error": {"units": "kg m-2"}},
            )
        assert not (tmp_path / "wrong.nc").exists()

    def test_default_metadata_keeps_observed_factors_visible(self, tmp_path):
        import netCDF4
        checks = {
            "respiration.q10_factor": 0.25,
            "soil.temperature_factor": 8.0,
            "soil.water_stress": 1.2,
        }
        path = tmp_path / "observed.nc"
        result = export_diagnostics_to_netcdf(
            [ConservationSummary(timestamp=1.0, custom_checks=checks, total_violations=0, tolerance_value=1e-6)], [1.0], path,
        )
        assert result.validation_passed
        with netCDF4.Dataset(path) as dataset:
            for key, value in checks.items():
                variable = dataset[key.replace('.', '_')]
                assert not np.ma.is_masked(variable[:])
                assert float(variable[0]) == pytest.approx(value)

    @pytest.mark.parametrize("second_unit", ["K", "Pa"])
    def test_diagnostic_identity_is_consistent_across_summaries(self, tmp_path, second_unit):
        summaries = [
            ConservationSummary(timestamp=0.0, custom_checks={"a.b": 1.0}, total_violations=0, tolerance_value=1e-6),
            ConservationSummary(timestamp=1.0, custom_checks={"a_b": 2.0}, total_violations=0, tolerance_value=1e-6),
        ]
        metadata = {"a.b": {"units": "K"}, "a_b": {"units": second_unit}}
        with pytest.raises(ValueError, match="diagnostic|Diagnostic|collision|attrs"):
            export_diagnostics_to_netcdf(summaries, [0.0, 1.0], tmp_path / "collision.nc", metadata_registry=metadata)
        assert not (tmp_path / "collision.nc").exists()

    def test_delivery_key_replays_original_metadata_and_receipt(self, tmp_path):
        summaries = [ConservationSummary(timestamp=1.0, balance_error={"C": 0.0002}, total_violations=0, tolerance_value=1e-6)]
        metadata = {"carbon_balance_error": {
            "units": "g m-2", "element": "C", "long_name": "Frozen carbon residual",
        }}
        path = tmp_path / "delivery.nc"
        first = export_diagnostics_to_netcdf(
            summaries, [1.0], path, delivery_key="step:1",
            metadata_registry=metadata,
        )
        before = path.read_bytes()
        second = export_diagnostics_to_netcdf(
            summaries, [1.0], path, delivery_key="step:1",
            metadata_registry=metadata,
        )
        assert first.validation_passed and second.replayed
        assert first.sha256 == second.sha256
        assert path.read_bytes() == before
        with xr.open_dataset(path) as dataset:
            assert dataset.carbon_balance_error.item() == pytest.approx(0.0002)
            assert dataset.carbon_balance_error.attrs["long_name"] == "Frozen carbon residual"
            assert dataset.attrs["history"].count("Processed by RELIC") == 1

    @pytest.mark.parametrize("checks", [
        {"invalid": {"nested": 1.0}},
        {"invalid": torch.tensor([1.0, 2.0])},
        {"same.name": 1.0, "same_name": 2.0},
        {"conservation_violations": 9.0},
    ])
    def test_unrepresentable_diagnostics_are_not_silently_omitted(self, tmp_path, checks):
        with pytest.raises((TypeError, ValueError), match="diagnostic|Diagnostic|scalar|collision"):
            export_diagnostics_to_netcdf(
                [ConservationSummary(timestamp=1.0, custom_checks=checks, total_violations=0, tolerance_value=1e-6)],
                [1.0], tmp_path / "invalid.nc",
            )
        assert not (tmp_path / "invalid.nc").exists()

    def test_convenience_export_function(self, multiple_summaries, tmp_path):
        """Test direct export to NetCDF using convenience function."""
        from relic.data.io.writer import WriteResult

        output_file = tmp_path / "diagnostics_test.nc"
        time_coords = [summary.timestamp for summary in multiple_summaries]
        
        result = export_diagnostics_to_netcdf(
            summaries=multiple_summaries,
            time_coords=time_coords,
            output_path=output_file,
            reference_date="2000-01-01"
        )
        
        # Check file was created
        assert output_file.exists()
        assert isinstance(result, WriteResult)
        assert result.output_path == output_file
        assert result.validation_passed is True
        assert result.cf_compliant is False
        assert result.file_size > 0
        
        # Load and verify
        with xr.open_dataset(output_file) as loaded_ds:
            assert 'time' in loaded_ds.dims
            np.testing.assert_allclose(
                loaded_ds['carbon_balance_error'].values,
                [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
            )

    def test_failed_write_receipt_is_not_reported_as_exported(
        self, multiple_summaries, tmp_path, monkeypatch, caplog,
    ):
        from relic.data.io.writer import DataWriter

        caplog.set_level("INFO", logger="relic.data.io.diagnostic_writer")

        def fail_write(self, *args, **kwargs):
            raise OSError("injected diagnostic disk failure")

        monkeypatch.setattr(DataWriter, "_write_with_format", fail_write)
        output_file = tmp_path / "failed.nc"
        # The writer raises the failure itself (no warning-and-receipt step in between).
        with pytest.raises(OSError, match="injected diagnostic disk failure"):
            export_diagnostics_to_netcdf(
                multiple_summaries,
                [summary.timestamp for summary in multiple_summaries],
                output_file,
            )
        assert not output_file.exists()
        assert "Exported diagnostics to" not in caplog.text

    def test_single_diagnostic_label_does_not_invent_interval_bounds(self, tmp_path):
        output_file = tmp_path / "first_step.nc"
        result = export_diagnostics_to_netcdf(
            [ConservationSummary(timestamp=1.0, balance_error={"C": 0.0002}, total_violations=0, tolerance_value=1e-6)],
            [1.0],
            output_file,
            reference_date="2000-01-01",
        )

        assert result.validation_passed is True
        with xr.open_dataset(output_file, decode_times=False) as dataset:
            assert dataset["time"].item() == pytest.approx(1.0 / float(SECONDS_PER_DAY))
            assert dataset["carbon_balance_error"].item() == pytest.approx(0.0002)
            assert "bounds" not in dataset["time"].attrs
            assert "time_bounds" not in dataset


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


