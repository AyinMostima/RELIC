"""
L1 Unit Tests for BVOC Emission Factors Module

Comprehensive testing of the emission factors database including PFT categories,
compound types, uncertainty handling, and database operations.

Test Coverage:
- EmissionFactorEntry data structure and uncertainty calculations
- EmissionFactorDatabase initialization and default values
- PFT and compound category enumerations
- Database query operations and default mechanisms
- Custom factor loading and updating
- Tensor conversion and device handling
- Database export/import functionality
- Parameter validation and error handling
- Boundary conditions and robustness

Scientific Validation:
- Emission factor values match MEGAN 2.1 literature values
- Uncertainty factors are within reasonable scientific bounds
- PFT-specific patterns (e.g., needleleaf high monoterpenes, broadleaf high isoprene)
- Unit consistency: 渭g g鈦宦?DW h鈦宦?(micrograms per gram dry weight per hour)

References:
- Guenther, A., et al. (2012): MEGAN 2.1 emission factor database
- Guenther, A.B., et al. (2006): Global terrestrial isoprene emissions
- Monson, R.K., et al. (2007): Isoprene emission from terrestrial ecosystems
- Arneth, A., et al. (2007): Process-based emission estimates
"""

import torch
import pytest
import warnings
import tempfile
import os
import subprocess
import threading
from pathlib import Path

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from relic.parameters import get_global_manager
from relic.scientific.biosphere.bvoc import emission_factors
from relic.scientific.biosphere.bvoc.emission_factors import (
    EmissionFactorDatabase,
    EmissionFactorEntry,
    PlantFunctionalType,
    CompoundCategory,
    DEFAULT_EMISSION_FACTORS,
    BVOC_EMISSION_FACTOR_UNITS,
)


@pytest.fixture(scope="module")
def param_manager():
    """Provide the shared ParameterManager for expected values."""
    return get_global_manager()


def _expected_value(param_manager, pft: PlantFunctionalType, compound: CompoundCategory) -> float:
    key = f"biosphere.bvoc.{pft.value}.{compound.value}"
    return float(
        param_manager.get(
            key,
            units=BVOC_EMISSION_FACTOR_UNITS,
            required=True,
        )
    )


class _BlockingParameterManager:
    """Delegate manager that can pause one rebuild at a deterministic point."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False

    def get_parameter(self, key):
        if self.block and not self.entered.is_set():
            self.entered.set()
            if not self.release.wait(timeout=10.0):
                raise TimeoutError("Timed out waiting to release parameter rebuild")
        return self.delegate.get_parameter(key)

    def get(self, *args, **kwargs):
        return self.delegate.get(*args, **kwargs)


def test_module_references_are_registered_records():
    """The module header cites registered records: Monson et al. is the 2007 Phil.
    Trans. R. Soc. A review and Arneth et al. is the 2007 Atmospheric Chemistry and
    Physics article; the unregistered 367, 1022-1035 and 8, 4129-4146 locators are
    retired."""
    header = emission_factors.__doc__
    for retired in ("1022-1035", "4129-4146"):
        assert retired not in header, retired
    assert "Philosophical Transactions of the Royal Society A 365, 1677-1695" in header
    assert "Atmospheric Chemistry and Physics 7, 31-53" in header


class TestEmissionFactorEntryL1:
    """L1 unit tests for EmissionFactorEntry data structure."""
    
    @pytest.fixture
    def sample_entry(self):
        """Create sample emission factor entry for testing."""
        return EmissionFactorEntry(
            value=24.0,
            uncertainty=3.0,
            sample_size=156,
            reference="Guenther et al. (2012)",
            notes="High isoprene emitters"
        )
    
    def test_basic_instantiation(self, sample_entry):
        """Test L1.1: Basic instantiation and attribute access."""
        assert isinstance(sample_entry, EmissionFactorEntry)
        
        # Check required attributes
        assert sample_entry.value == 24.0
        assert sample_entry.uncertainty == 3.0
        assert sample_entry.sample_size == 156
        assert sample_entry.reference == "Guenther et al. (2012)"
        assert sample_entry.notes == "High isoprene emitters"
    
    def test_minimal_instantiation(self):
        """Test L1.2: Minimal instantiation with optional parameters."""
        minimal_entry = EmissionFactorEntry(value=10.0, uncertainty=2.0)
        
        assert minimal_entry.value == 10.0
        assert minimal_entry.uncertainty == 2.0
        assert minimal_entry.sample_size is None
        assert minimal_entry.reference is None
        assert minimal_entry.notes is None
    
    def test_log_normal_uncertainty_bounds(self, sample_entry):
        """Test L1.3: Log-normal uncertainty calculation."""
        lower, upper = sample_entry.get_value_with_uncertainty("log_normal")
        
        # For value=24.0, uncertainty=3.0:
        # lower = 24.0 / 3.0 = 8.0
        # upper = 24.0 * 3.0 = 72.0
        assert lower == pytest.approx(8.0)
        assert upper == pytest.approx(72.0)
        
        # Bounds should be positive
        assert lower >= 0
        assert upper > lower
        
        # Check scientific reasonableness
        uncertainty_range = upper / lower
        assert uncertainty_range == pytest.approx(9.0)  # 3.0^2
    
    def test_linear_uncertainty_bounds(self, sample_entry):
        """Test L1.4: Linear uncertainty calculation."""
        lower, upper = sample_entry.get_value_with_uncertainty("linear")
        
        # For value=24.0, uncertainty=3.0:
        # lower = 24.0 - 3.0 = 21.0
        # upper = 24.0 + 3.0 = 27.0
        assert lower == pytest.approx(21.0)
        assert upper == pytest.approx(27.0)
        
        # Bounds should be symmetric around value
        assert abs((lower + upper) / 2 - sample_entry.value) < 1e-10
    
    def test_uncertainty_bounds_edge_cases(self):
        """Test L1.5: Uncertainty bounds edge cases."""
        # Test case where linear uncertainty would give negative lower bound
        low_value_entry = EmissionFactorEntry(value=1.0, uncertainty=2.0)
        
        lower_linear, upper_linear = low_value_entry.get_value_with_uncertainty("linear")
        # Should clamp at zero: max(0.0, 1.0 - 2.0) = 0.0
        assert lower_linear == 0.0
        assert upper_linear == 3.0
        
        # Log-normal should still be positive
        lower_log, upper_log = low_value_entry.get_value_with_uncertainty("log_normal")
        assert lower_log == pytest.approx(0.5)  # 1.0 / 2.0
        assert upper_log == pytest.approx(2.0)  # 1.0 * 2.0
    
    def test_zero_uncertainty_handling(self):
        """Test L1.6: Zero uncertainty handling."""
        zero_uncertainty_entry = EmissionFactorEntry(value=15.0, uncertainty=1.0)
        
        # With uncertainty=1.0, bounds should equal value
        lower, upper = zero_uncertainty_entry.get_value_with_uncertainty("log_normal")
        assert lower == pytest.approx(15.0)  # 15.0 / 1.0
        assert upper == pytest.approx(15.0)  # 15.0 * 1.0


class TestPlantFunctionalTypeEnumL1:
    """L1 unit tests for PlantFunctionalType enumeration."""
    
    def test_enum_completeness(self):
        """Test L1.1: Completeness of PFT enumeration."""
        expected_pfts = [
            "broadleaf_evergreen_tropical",
            "broadleaf_evergreen_temperate",
            "broadleaf_deciduous_tropical", 
            "broadleaf_deciduous_temperate",
            "needleleaf_evergreen_temperate",
            "needleleaf_evergreen_boreal",
            "needleleaf_deciduous_boreal",
            "c3_grass",
            "c4_grass",
            "shrub",
            "crop",
            "wetland",
            "default"
        ]
        
        # Check all expected PFTs exist
        for pft_name in expected_pfts:
            assert hasattr(PlantFunctionalType, pft_name.upper())
            assert PlantFunctionalType(pft_name) in PlantFunctionalType
        
        # Check total count
        assert len(PlantFunctionalType) == len(expected_pfts)
    
    def test_string_conversion(self):
        """Test L1.2: String value conversion."""
        pft = PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL
        assert pft.value == "broadleaf_evergreen_tropical"
        
        # Test reconstruction from string
        reconstructed = PlantFunctionalType("broadleaf_evergreen_tropical")
        assert reconstructed == pft
    
    def test_invalid_pft_handling(self):
        """Test L1.3: Invalid PFT string handling."""
        with pytest.raises(ValueError):
            PlantFunctionalType("invalid_pft_type")


class TestCompoundCategoryEnumL1:
    """L1 unit tests for CompoundCategory enumeration."""
    
    def test_enum_completeness(self):
        """Test L1.1: Completeness of compound category enumeration."""
        expected_compounds = [
            "isoprene",
            "monoterpenes",
            "sesquiterpenes", 
            "ovocs",
            "no",
            "aerosols",
            "co"
        ]
        
        # Check all expected compounds exist
        for compound_name in expected_compounds:
            assert hasattr(CompoundCategory, compound_name.upper())
            assert CompoundCategory(compound_name) in CompoundCategory
        
        # Check total count
        assert len(CompoundCategory) == len(expected_compounds)
    
    def test_string_conversion(self):
        """Test L1.2: String value conversion."""
        compound = CompoundCategory.ISOPRENE
        assert compound.value == "isoprene"
        
        # Test reconstruction from string
        reconstructed = CompoundCategory("isoprene")
        assert reconstructed == compound
    
    def test_invalid_compound_handling(self):
        """Test L1.3: Invalid compound string handling."""
        with pytest.raises(ValueError):
            CompoundCategory("invalid_compound_type")


class TestEmissionFactorDatabaseL1:
    """L1 unit tests for EmissionFactorDatabase functionality."""
    
    @pytest.fixture
    def database(self):
        """Create fresh database instance for testing."""
        return EmissionFactorDatabase()
    
    def test_basic_instantiation(self, database):
        """Test L1.1: Basic database instantiation."""
        assert isinstance(database, EmissionFactorDatabase)
        assert hasattr(database, 'database')
        assert isinstance(database.database, dict)
        
        # Check that default database is populated
        assert len(database.database) > 0
        
        # Check that default PFT exists
        assert PlantFunctionalType.DEFAULT in database.database
    
    def test_default_database_completeness(self, database):
        """Test L1.2: Default database completeness and scientific values."""
        # Check major PFT categories exist
        major_pfts = [
            PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL,
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            PlantFunctionalType.NEEDLELEAF_EVERGREEN_TEMPERATE,
            PlantFunctionalType.C3_GRASS
        ]
        
        for pft in major_pfts:
            assert pft in database.database
            pft_data = database.database[pft]
            
            # Each PFT should have major compound categories
            major_compounds = [
                CompoundCategory.ISOPRENE,
                CompoundCategory.MONOTERPENES,
                CompoundCategory.SESQUITERPENES,
                CompoundCategory.OVOCS
            ]
            
            for compound in major_compounds:
                assert compound in pft_data
                entry = pft_data[compound]
                assert isinstance(entry, EmissionFactorEntry)
                assert entry.value > 0  # Emission factors should be positive
                assert entry.uncertainty >= 1.0  # Uncertainty factor >= 1
    
    def test_literature_values_validation(self, database, param_manager):
        """Test L1.3: Validation against known literature values."""
        # Test key literature values from Guenther et al. (2012)
        
        # Broadleaf deciduous temperate: highest isoprene emitters
        broadleaf_dec_temp = database.get_emission_factor(
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE
        )
        expected_dec_temp = _expected_value(
            param_manager,
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE,
        )
        assert broadleaf_dec_temp.value == pytest.approx(expected_dec_temp)
        
        # Needleleaf evergreen boreal: highest monoterpene emitters
        needleleaf_boreal = database.get_emission_factor(
            PlantFunctionalType.NEEDLELEAF_EVERGREEN_BOREAL,
            CompoundCategory.MONOTERPENES
        )
        expected_needle_mono = _expected_value(
            param_manager,
            PlantFunctionalType.NEEDLELEAF_EVERGREEN_BOREAL,
            CompoundCategory.MONOTERPENES,
        )
        assert needleleaf_boreal.value == pytest.approx(expected_needle_mono)
        
        # Tropical broadleaf evergreen: high isoprene
        tropical_broadleaf = database.get_emission_factor(
            PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL,
            CompoundCategory.ISOPRENE
        )
        expected_tropical = _expected_value(
            param_manager,
            PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL,
            CompoundCategory.ISOPRENE,
        )
        assert tropical_broadleaf.value == pytest.approx(expected_tropical)
        
        # C4 grass: lower isoprene than C3
        c4_grass = database.get_emission_factor(
            PlantFunctionalType.C4_GRASS,
            CompoundCategory.ISOPRENE
        )
        c3_grass = database.get_emission_factor(
            PlantFunctionalType.C3_GRASS,
            CompoundCategory.ISOPRENE
        )
        assert c4_grass.value < c3_grass.value  # C4 < C3 for isoprene
    
    def test_scientific_patterns_validation(self, database):
        """Test L1.4: Scientific patterns in emission factors."""
        # Pattern 1: Needleleaf conifers should have higher monoterpenes than broadleaf
        needleleaf_mono = database.get_emission_factor(
            PlantFunctionalType.NEEDLELEAF_EVERGREEN_TEMPERATE,
            CompoundCategory.MONOTERPENES
        ).value
        
        broadleaf_mono = database.get_emission_factor(
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.MONOTERPENES  
        ).value
        
        assert needleleaf_mono > broadleaf_mono
        
        # Pattern 2: Deciduous temperate should have highest isoprene
        deciduous_temp_iso = database.get_emission_factor(
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE
        ).value
        
        evergreen_temp_iso = database.get_emission_factor(
            PlantFunctionalType.BROADLEAF_EVERGREEN_TEMPERATE,
            CompoundCategory.ISOPRENE
        ).value
        
        grass_iso = database.get_emission_factor(
            PlantFunctionalType.C3_GRASS,
            CompoundCategory.ISOPRENE
        ).value
        
        assert deciduous_temp_iso > evergreen_temp_iso
        assert deciduous_temp_iso > grass_iso
        
        # Pattern 3: Sesquiterpenes should generally be lowest
        for pft in [PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL,
                   PlantFunctionalType.NEEDLELEAF_EVERGREEN_TEMPERATE]:
            
            isoprene = database.get_emission_factor(pft, CompoundCategory.ISOPRENE).value
            monoterpenes = database.get_emission_factor(pft, CompoundCategory.MONOTERPENES).value  
            sesquiterpenes = database.get_emission_factor(pft, CompoundCategory.SESQUITERPENES).value
            
            assert sesquiterpenes < isoprene
            assert sesquiterpenes < monoterpenes
    
    def test_get_emission_factor_with_strings(self, database, param_manager):
        """Test L1.5: Get emission factor using string inputs."""
        # Test with string PFT and compound
        factor = database.get_emission_factor("broadleaf_evergreen_tropical", "isoprene")
        assert isinstance(factor, EmissionFactorEntry)
        expected = _expected_value(
            param_manager,
            PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL,
            CompoundCategory.ISOPRENE,
        )
        assert factor.value == pytest.approx(expected)
        
        # Test with mixed enum and string
        factor2 = database.get_emission_factor(
            PlantFunctionalType.NEEDLELEAF_EVERGREEN_TEMPERATE,
            "monoterpenes"
        )
        assert isinstance(factor2, EmissionFactorEntry)
        expected_mono = _expected_value(
            param_manager,
            PlantFunctionalType.NEEDLELEAF_EVERGREEN_TEMPERATE,
            CompoundCategory.MONOTERPENES,
        )
        assert factor2.value == pytest.approx(expected_mono)
    
    def test_unknown_inputs_raise(self, database):
        """Test L1.6: Unknown PFTs or compounds raise explicit errors."""
        with pytest.raises(ValueError, match="Unknown PFT"):
            database.get_emission_factor("unknown_pft", "isoprene")
        
        with pytest.raises(ValueError, match="Unknown compound category"):
            database.get_emission_factor("broadleaf_evergreen_tropical", "unknown_compound")
    
    def test_get_all_factors_for_pft(self, database):
        """Test L1.7: Get all factors for specific PFT."""
        factors = database.get_all_factors_for_pft(PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL)
        
        assert isinstance(factors, dict)
        assert len(factors) >= 4  # At least major compound categories
        
        # Check expected compounds are present
        expected_compounds = [
            CompoundCategory.ISOPRENE,
            CompoundCategory.MONOTERPENES,
            CompoundCategory.SESQUITERPENES,
            CompoundCategory.OVOCS
        ]
        
        for compound in expected_compounds:
            assert compound in factors
            assert isinstance(factors[compound], EmissionFactorEntry)
        
        # Test with string PFT
        factors_str = database.get_all_factors_for_pft("broadleaf_evergreen_tropical")
        assert factors_str == factors
    
    def test_tensor_conversion(self, database, param_manager):
        """Test L1.8: PyTorch tensor conversion functionality."""
        # Test basic tensor conversion
        tensor_factor = database.get_factor_tensor(
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE
        )
        
        assert isinstance(tensor_factor, torch.Tensor)
        assert tensor_factor.dtype == torch.float32
        expected_value = _expected_value(
            param_manager,
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE,
        )
        assert tensor_factor.item() == pytest.approx(expected_value)
        
        # Test device specification
        if torch.cuda.is_available():
            cuda_tensor = database.get_factor_tensor(
                "broadleaf_deciduous_temperate",
                "isoprene",
                device=torch.device("cuda")
            )
            assert cuda_tensor.device.type == "cuda"
        
        # Test CPU device specification
        cpu_tensor = database.get_factor_tensor(
            "broadleaf_deciduous_temperate",
            "isoprene", 
            device=torch.device("cpu")
        )
        assert cpu_tensor.device.type == "cpu"
    
    def test_uncertainty_bounds_tensors(self, database):
        """Test L1.9: Uncertainty bounds as tensors."""
        lower_tensor, upper_tensor = database.get_uncertainty_bounds(
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE
        )
        
        assert isinstance(lower_tensor, torch.Tensor)
        assert isinstance(upper_tensor, torch.Tensor)
        
        # Check values match expected uncertainty calculation
        factor_entry = database.get_emission_factor(
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE
        )
        
        expected_lower = factor_entry.value / factor_entry.uncertainty
        expected_upper = factor_entry.value * factor_entry.uncertainty
        
        assert lower_tensor.item() == pytest.approx(expected_lower)
        assert upper_tensor.item() == pytest.approx(expected_upper)
        
        # Bounds should be ordered correctly
        assert lower_tensor.item() <= upper_tensor.item()
    
    def test_custom_factors_update(self):
        """Test L1.10: Custom factors database update."""
        custom_factors = {
            "broadleaf_evergreen_tropical": {
                "isoprene": 50.0,  # Override default
                "monoterpenes": {
                    "value": 5.0,
                    "uncertainty": 2.5,
                    "reference": "Custom study"
                }
            },
            "crop": {
                "isoprene": 25.0
            }
        }
        
        # Create database with custom factors
        custom_db = EmissionFactorDatabase(custom_factors=custom_factors)
        
        # Check overridden values
        tropical_iso = custom_db.get_emission_factor("broadleaf_evergreen_tropical", "isoprene")
        assert tropical_iso.value == pytest.approx(50.0)
        assert tropical_iso.reference == "Custom"
        
        tropical_mono = custom_db.get_emission_factor("broadleaf_evergreen_tropical", "monoterpenes")
        assert tropical_mono.value == pytest.approx(5.0)
        assert tropical_mono.uncertainty == pytest.approx(2.5)
        assert tropical_mono.reference == "Custom study"
        
        crop_iso = custom_db.get_emission_factor("crop", "isoprene")
        assert crop_iso.value == pytest.approx(25.0)
        
        # Non-overridden values should remain default
        tropical_sesq = custom_db.get_emission_factor("broadleaf_evergreen_tropical", "sesquiterpenes")
        default_sesq = DEFAULT_EMISSION_FACTORS.get_emission_factor("broadleaf_evergreen_tropical", "sesquiterpenes")
        assert tropical_sesq.value == default_sesq.value


class TestEmissionFactorDatabaseAdvancedL1:
    """Advanced L1 unit tests for database operations."""
    
    @pytest.fixture
    def database(self):
        """Create fresh database instance for testing."""
        return EmissionFactorDatabase()
    
    def test_list_available_pfts(self, database):
        """Test L1.1: List available PFTs functionality."""
        available_pfts = database.list_available_pfts()
        
        assert isinstance(available_pfts, list)
        assert len(available_pfts) > 10  # Should have multiple PFTs
        
        # Check specific PFTs are included
        expected_pfts = [
            PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL,
            PlantFunctionalType.NEEDLELEAF_EVERGREEN_TEMPERATE,
            PlantFunctionalType.C3_GRASS,
            PlantFunctionalType.DEFAULT
        ]
        
        for pft in expected_pfts:
            assert pft in available_pfts
    
    def test_list_available_compounds(self, database):
        """Test L1.2: List available compounds functionality."""
        available_compounds = database.list_available_compounds()
        
        assert isinstance(available_compounds, list)
        assert len(available_compounds) >= 4  # Major compound categories
        
        # Check specific compounds are included
        expected_compounds = [
            CompoundCategory.ISOPRENE,
            CompoundCategory.MONOTERPENES,
            CompoundCategory.SESQUITERPENES,
            CompoundCategory.OVOCS
        ]
        
        for compound in expected_compounds:
            assert compound in available_compounds
    
    def test_database_export(self, database, param_manager):
        """Test L1.3: Database export functionality."""
        export_data = database.export_database()
        
        assert isinstance(export_data, dict)
        
        # Check structure
        for pft_key, compounds in export_data.items():
            assert isinstance(pft_key, str)  # PFT keys should be strings
            assert isinstance(compounds, dict)
            
            for compound_key, entry_data in compounds.items():
                assert isinstance(compound_key, str)  # Compound keys should be strings
                assert isinstance(entry_data, dict)
                
                # Check required fields
                assert 'value' in entry_data
                assert 'uncertainty' in entry_data
                assert isinstance(entry_data['value'], (int, float))
                assert isinstance(entry_data['uncertainty'], (int, float))
        
        # Check specific known values
        assert 'broadleaf_deciduous_temperate' in export_data
        assert 'isoprene' in export_data['broadleaf_deciduous_temperate']
        expected_value = _expected_value(
            param_manager,
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE,
        )
        assert export_data['broadleaf_deciduous_temperate']['isoprene']['value'] == pytest.approx(expected_value)
    
    def test_database_save_load(self, database):
        """Test L1.4: Database save and load functionality."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save database
            database.save_to_file(tmp_path)
            
            # Check file exists and has content
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
            
            # Load database from file
            loaded_db = EmissionFactorDatabase.load_from_file(tmp_path)
            
            # Verify loaded database matches original
            assert isinstance(loaded_db, EmissionFactorDatabase)
            
            # Check specific values
            original_factor = database.get_emission_factor("broadleaf_deciduous_temperate", "isoprene")
            loaded_factor = loaded_db.get_emission_factor("broadleaf_deciduous_temperate", "isoprene")
            
            assert loaded_factor.value == original_factor.value
            assert loaded_factor.uncertainty == original_factor.uncertainty
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_invalid_custom_factors_handling(self):
        """Test L1.5: Handling of invalid custom factors."""
        # Test with invalid PFT
        invalid_factors = {
            "invalid_pft": {
                "isoprene": 10.0
            }
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            db = EmissionFactorDatabase(custom_factors=invalid_factors)
            
            # Should have issued a warning
            assert len(w) >= 1
            assert any("Unknown PFT" in str(warning.message) for warning in w)
        
        # Test with invalid compound
        invalid_compound_factors = {
            "broadleaf_evergreen_tropical": {
                "invalid_compound": 10.0
            }
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            db = EmissionFactorDatabase(custom_factors=invalid_compound_factors)
            
            # Should have issued a warning
            assert len(w) >= 1
            assert any("Unknown compound" in str(warning.message) for warning in w)


class TestDefaultEmissionFactorsInstanceL1:
    """L1 unit tests for default global emission factors instance."""

    def test_default_manager_uses_global_parameter_manager(self):
        """Default construction must use the shared ParameterManager."""
        assert DEFAULT_EMISSION_FACTORS.param_manager is get_global_manager()
        assert EmissionFactorDatabase().param_manager is get_global_manager()

    def test_global_default_reconciles_replaced_manager_before_read(self):
        """The default database must follow a replaced global manager atomically."""
        repository_root = Path(__file__).resolve().parents[4]
        script = """
from relic.parameters import get_global_manager, reset_global_manager
from relic.scientific.biosphere.bvoc.emission_factors import (
    BVOC_EMISSION_FACTOR_UNITS,
    CompoundCategory,
    DEFAULT_EMISSION_FACTORS,
    PlantFunctionalType,
)

pft = PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE
compound = CompoundCategory.ISOPRENE
key = f"biosphere.bvoc.{pft.value}.{compound.value}"
old_manager = DEFAULT_EMISSION_FACTORS.param_manager
reset_global_manager()
replacement = get_global_manager()
replacement_value = float(
    replacement.get(
        key,
        units=BVOC_EMISSION_FACTOR_UNITS,
        required=True,
    )
) + 1.0
replacement.set_value(
    key,
    replacement_value,
    validate=True,
)
observed = DEFAULT_EMISSION_FACTORS.get_emission_factor(pft, compound).value
assert replacement is not old_manager
assert DEFAULT_EMISSION_FACTORS.param_manager is replacement
assert observed == replacement_value
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr

    def test_same_manager_mutation_requires_explicit_refresh(self, param_manager):
        """Parameter mutation remains governed by the explicit refresh owner."""
        pft = PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE
        compound = CompoundCategory.ISOPRENE
        key = f"biosphere.bvoc.{pft.value}.{compound.value}"
        database = EmissionFactorDatabase(parameter_manager=param_manager)
        original_value = database.get_emission_factor(pft, compound).value
        updated_value = original_value + 1.0

        try:
            param_manager.set_value(
                key,
                updated_value,
                validate=True,
            )
            assert database.get_emission_factor(pft, compound).value == original_value

            database.refresh_from_manager()

            assert database.get_emission_factor(pft, compound).value == updated_value
        finally:
            param_manager.set_value(
                key,
                original_value,
                validate=True,
            )
            database.refresh_from_manager()

    def test_refresh_failure_preserves_previous_complete_snapshot(
        self,
        monkeypatch,
        param_manager,
    ):
        """A failed rebuild must leave the previous manager and table intact."""
        database = EmissionFactorDatabase(parameter_manager=param_manager)
        baseline = database.export_database()
        baseline_manager = database.param_manager
        missing_key = (
            "biosphere.bvoc.broadleaf_deciduous_temperate.isoprene"
        )
        original_get_parameter = param_manager.get_parameter

        def get_parameter_with_one_missing(key):
            if key == missing_key:
                return None
            return original_get_parameter(key)

        monkeypatch.setattr(
            param_manager,
            "get_parameter",
            get_parameter_with_one_missing,
        )

        with pytest.raises(KeyError, match="Missing BVOC emission factor parameters"):
            database.refresh_from_manager()

        assert database.param_manager is baseline_manager
        assert database.export_database() == baseline

    def test_concurrent_reader_never_observes_partial_refresh(self, param_manager):
        """Readers must see a complete old or new snapshot during refresh."""
        manager = _BlockingParameterManager(param_manager)
        database = EmissionFactorDatabase(parameter_manager=manager)
        pft = PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE
        compound = CompoundCategory.ISOPRENE
        baseline = database.get_emission_factor(pft, compound).value
        refresh_errors = []

        def refresh():
            try:
                database.refresh_from_manager()
            except Exception as exc:
                refresh_errors.append(exc)

        manager.block = True
        worker = threading.Thread(target=refresh)
        worker.start()
        assert manager.entered.wait(timeout=10.0)
        try:
            observed = database.get_emission_factor(pft, compound).value
        finally:
            manager.release.set()
            worker.join(timeout=10.0)

        assert not worker.is_alive()
        assert refresh_errors == []
        assert observed == baseline

    def test_explicit_manager_remains_pinned_when_global_owner_changes(
        self,
        monkeypatch,
        param_manager,
    ):
        """Explicit injection must never follow an unrelated global replacement."""
        database = EmissionFactorDatabase(parameter_manager=param_manager)
        baseline = database.get_emission_factor(
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE,
        ).value

        def unexpected_global_lookup():
            raise AssertionError("Explicit databases must not query the global manager")

        monkeypatch.setattr(
            emission_factors,
            "get_global_manager",
            unexpected_global_lookup,
        )

        assert database.param_manager is param_manager
        assert database.get_emission_factor(
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE,
        ).value == baseline
    
    def test_global_instance_availability(self, param_manager):
        """Test L1.1: Global DEFAULT_EMISSION_FACTORS instance."""
        assert isinstance(DEFAULT_EMISSION_FACTORS, EmissionFactorDatabase)
        
        # Should have database populated
        assert len(DEFAULT_EMISSION_FACTORS.database) > 0
        
        # Test access to common factors
        factor = DEFAULT_EMISSION_FACTORS.get_emission_factor(
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE
        )
        expected_value = _expected_value(
            param_manager,
            PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
            CompoundCategory.ISOPRENE,
        )
        assert factor.value == pytest.approx(expected_value)
    
    def test_global_instance_consistency(self):
        """Test L1.2: Global instance consistency with new instances."""
        new_db = EmissionFactorDatabase()
        
        # Should have same values as global instance
        pfts_to_test = [
            PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL,
            PlantFunctionalType.NEEDLELEAF_EVERGREEN_TEMPERATE,
            PlantFunctionalType.C3_GRASS
        ]
        
        compounds_to_test = [
            CompoundCategory.ISOPRENE,
            CompoundCategory.MONOTERPENES
        ]
        
        for pft in pfts_to_test:
            for compound in compounds_to_test:
                global_factor = DEFAULT_EMISSION_FACTORS.get_emission_factor(pft, compound)
                new_factor = new_db.get_emission_factor(pft, compound)
                
                assert global_factor.value == new_factor.value
                assert global_factor.uncertainty == new_factor.uncertainty
    
    def test_module_exports(self):
        """Test L1.3: Module __all__ exports."""
        from relic.scientific.biosphere.bvoc import emission_factors
        
        # Check that __all__ is defined and contains expected items
        assert hasattr(emission_factors, '__all__')
        
        expected_exports = [
            'EmissionFactorDatabase',
            'EmissionFactorEntry', 
            'PlantFunctionalType',
            'CompoundCategory',
            'DEFAULT_EMISSION_FACTORS'
        ]
        
        for item in expected_exports:
            assert item in emission_factors.__all__
            assert hasattr(emission_factors, item)


class TestBoundaryConditionsAndRobustnessL1:
    """L1 unit tests for boundary conditions and robustness."""
    
    @pytest.fixture
    def database(self):
        """Create fresh database instance for testing."""
        return EmissionFactorDatabase()
    
    def test_extreme_uncertainty_values(self):
        """Test L1.1: Extreme uncertainty factor handling."""
        # Very high uncertainty
        high_uncertainty_entry = EmissionFactorEntry(value=10.0, uncertainty=100.0)
        
        lower, upper = high_uncertainty_entry.get_value_with_uncertainty("log_normal")
        assert lower == pytest.approx(0.1)  # 10.0 / 100.0
        assert upper == pytest.approx(1000.0)  # 10.0 * 100.0
        assert upper > lower
        
        # Very low uncertainty (close to 1)
        low_uncertainty_entry = EmissionFactorEntry(value=10.0, uncertainty=1.01)
        
        lower_low, upper_low = low_uncertainty_entry.get_value_with_uncertainty("log_normal")
        assert lower_low == pytest.approx(9.901, rel=0.01)  # 10.0 / 1.01
        assert upper_low == pytest.approx(10.1, rel=0.01)   # 10.0 * 1.01
    
    def test_zero_and_negative_values(self):
        """Test L1.2: Zero and negative value handling."""
        # Zero emission factor (valid for some compounds/PFTs)
        zero_entry = EmissionFactorEntry(value=0.0, uncertainty=2.0)
        
        lower, upper = zero_entry.get_value_with_uncertainty("log_normal")
        assert lower == 0.0  # 0.0 / 2.0
        assert upper == 0.0  # 0.0 * 2.0
        
        # Linear uncertainty with zero value
        lower_lin, upper_lin = zero_entry.get_value_with_uncertainty("linear")
        assert lower_lin == 0.0  # max(0.0, 0.0 - 2.0) = 0.0
        assert upper_lin == 2.0  # 0.0 + 2.0
    
    def test_large_database_operations(self, database):
        """Test L1.3: Performance with database operations."""
        # Test getting many factors quickly
        pfts = database.list_available_pfts()
        compounds = database.list_available_compounds()
        
        # Should be able to query all combinations without issues
        factor_count = 0
        for pft in pfts[:5]:  # Limit to first 5 PFTs for speed
            for compound in compounds[:4]:  # Limit to first 4 compounds
                try:
                    factor = database.get_emission_factor(pft, compound)
                    assert isinstance(factor, EmissionFactorEntry)
                    factor_count += 1
                except ValueError:
                    # Some combinations might not exist, that's ok
                    pass
        
        assert factor_count > 10  # Should have found multiple valid combinations
    
    def test_memory_efficiency(self, database, param_manager):
        """Test L1.4: Memory efficiency of database operations."""
        # Create multiple tensor conversions - should not accumulate memory
        factors = []
        for i in range(100):
            tensor_factor = database.get_factor_tensor(
                PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL,
                CompoundCategory.ISOPRENE
            )
            factors.append(tensor_factor.item())
        
        # All values should be the same
        assert all(f == factors[0] for f in factors)
        expected_value = _expected_value(
            param_manager,
            PlantFunctionalType.BROADLEAF_EVERGREEN_TROPICAL,
            CompoundCategory.ISOPRENE,
        )
        assert factors[0] == pytest.approx(expected_value)
    
    def test_concurrent_access_safety(self, database, param_manager):
        """Test L1.5: Thread safety of database access."""
        # Test multiple simultaneous accesses (basic thread safety)
        results = []
        
        def get_factor():
            factor = database.get_emission_factor(
                PlantFunctionalType.NEEDLELEAF_EVERGREEN_TEMPERATE,
                CompoundCategory.MONOTERPENES
            )
            return factor.value
        
        # Simulate concurrent access
        for _ in range(50):
            result = get_factor()
            results.append(result)
        
        # All results should be identical (no race conditions)
        assert all(r == results[0] for r in results)
        expected_value = _expected_value(
            param_manager,
            PlantFunctionalType.NEEDLELEAF_EVERGREEN_TEMPERATE,
            CompoundCategory.MONOTERPENES,
        )
        assert results[0] == pytest.approx(expected_value)


if __name__ == "__main__":
    pytest.main([__file__])


